// esp32_llm.ino
// "I made ChatGPT so dumb it ran on an ESP32"
// Architecture: 2-layer GPT, char-level, EMBED_SIZE=64, BLOCK_SIZE=64
// Weights live in flash (PROGMEM). Only activations use RAM.

#include <Arduino.h>
#include <pgmspace.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#include "weights.h"  // all model weights in PROGMEM
#include "vocab.h"    // ITOS[], STOI[]

// ─── LCD ─────────────────────────────────────────────────────────────────────
// SDA = GPIO21, SCL = GPIO22 on ESP32 DevKit V1
// Change 0x27 to 0x3F if your LCD doesn't light up
LiquidCrystal_I2C lcd(0x27, 16, 2);

// ─── Generation config ────────────────────────────────────────────────────────
#define MAX_NEW_TOKENS 60  // tokens to generate per prompt
#define TEMPERATURE 1.0f   // 1.0 = raw softmax, lower = more deterministic
#define TOP_K 5            // only sample from top-K logits (0 = disabled)

// ─── RAM buffers (kept minimal) ───────────────────────────────────────────────
// We never allocate EMBED_SIZE*BLOCK_SIZE for the full sequence in RAM.
// Instead we keep a token history ring and re-run forward pass each step.
static int16_t token_buf[BLOCK_SIZE];  // current context window
static int ctx_len = 0;

// Activation scratch — reused across layers.
// Largest single buffer needed: 4*EMBED_SIZE for MLP intermediate.
static float x[EMBED_SIZE];               // current token embedding (working vector)
static float x2[EMBED_SIZE];              // residual / scratch
static float attn_out[EMBED_SIZE];        // attention output
static float qkv[3 * EMBED_SIZE];         // Q, K, V for one token position
static float mlp_hidden[4 * EMBED_SIZE];  // MLP intermediate
static float logits[VOCAB_SIZE];          // final output logits

// KV cache: store K and V for all positions, all layers, all heads
// Size: NUM_LAYERS * BLOCK_SIZE * EMBED_SIZE * 2  (K and V)
// = 2 * 64 * 64 * 2 = 16 384 floats = 64 KB
// ESP32 has 520 KB SRAM so this is fine.
static float kv_cache_k[NUM_LAYERS][BLOCK_SIZE][EMBED_SIZE];
static float kv_cache_v[NUM_LAYERS][BLOCK_SIZE][EMBED_SIZE];

// ─── PROGMEM helpers ──────────────────────────────────────────────────────────
inline float pgm_float(const float* addr) {
  float v;
  memcpy_P(&v, addr, sizeof(float));
  return v;
}

// Read a row of length `cols` from a PROGMEM 2-D matrix (row-major)
// into a RAM buffer `out`.
void pgm_read_row(const float* base, int row, int cols, float* out) {
  memcpy_P(out, base + (size_t)row * cols, cols * sizeof(float));
}

// ─── Math helpers ─────────────────────────────────────────────────────────────
float relu(float x) {
  return x > 0 ? x : 0;
}

// GELU approximation (tanh-based, matches PyTorch)
float gelu(float x) {
  float x3 = x * x * x;
  float inner = 0.7978845608f * (x + 0.044715f * x3);
  // tanh approx: tanh(x) ≈ x*(27+x^2)/(27+9*x^2) — fast, good enough
  float t;
  if (inner > 4.0f) t = 1.0f;
  else if (inner < -4.0f) t = -1.0f;
  else {
    float i2 = inner * inner;
    t = inner * (27.0f + i2) / (27.0f + 9.0f * i2);
  }
  return 0.5f * x * (1.0f + t);
}

void softmax(float* v, int n) {
  float maxv = v[0];
  for (int i = 1; i < n; i++)
    if (v[i] > maxv) maxv = v[i];
  float sum = 0;
  for (int i = 0; i < n; i++) {
    v[i] = expf(v[i] - maxv);
    sum += v[i];
  }
  for (int i = 0; i < n; i++) v[i] /= sum;
}

// Layer norm: out = (x - mean) / sqrt(var + eps) * w + b
// w and b are PROGMEM pointers
void layer_norm(float* out, const float* in,
                const float* w_pgm, const float* b_pgm, int n) {
  float mean = 0, var = 0;
  for (int i = 0; i < n; i++) mean += in[i];
  mean /= n;
  for (int i = 0; i < n; i++) {
    float d = in[i] - mean;
    var += d * d;
  }
  var /= n;
  float inv = 1.0f / sqrtf(var + 1e-5f);
  for (int i = 0; i < n; i++) {
    out[i] = (in[i] - mean) * inv
               * pgm_float(w_pgm + i)
             + pgm_float(b_pgm + i);
  }
}

// Matrix-vector multiply: out[M] = W[M][N] * in[N] + bias[M]
// W is PROGMEM row-major. bias may be NULL.
void matvec(float* out, const float* W_pgm, const float* bias_pgm,
            int M, int N, const float* in) {
  for (int i = 0; i < M; i++) {
    float acc = 0;
    const float* row = W_pgm + (size_t)i * N;
    for (int j = 0; j < N; j++) acc += pgm_float(row + j) * in[j];
    out[i] = acc + (bias_pgm ? pgm_float(bias_pgm + i) : 0.0f);
  }
}

// ─── Transformer block forward pass ──────────────────────────────────────────
// Runs ONE block for the LAST token position only (autoregressive, no batch).
// Uses the KV cache so we don't recompute all past positions.
//
// layer_i  : which transformer block (0 or 1)
// pos      : current sequence position (0-based)
// in_vec   : input embedding [EMBED_SIZE]  (modified in place, residual)

struct LayerWeights {
  const float *ln1_w, *ln1_b;
  const float *ln2_w, *ln2_b;
  const float* attn_qkv_w;
  const float* attn_proj_w;
  const float *mlp_fc_w, *mlp_fc_b;
  const float *mlp_proj_w, *mlp_proj_b;
};

// Pointers to each layer's weights — filled in setup()
LayerWeights layer_wts[NUM_LAYERS];

void transformer_block(int layer_i, int pos, float* vec) {
  const LayerWeights& W = layer_wts[layer_i];

  // ── 1. LayerNorm 1 ────────────────────────────────────────────────────────
  layer_norm(x2, vec, W.ln1_w, W.ln1_b, EMBED_SIZE);

  // ── 2. Compute Q, K, V for this position ──────────────────────────────────
  // c_attn weight is [3*EMBED, EMBED], bias=False
  matvec(qkv, W.attn_qkv_w, nullptr, 3 * EMBED_SIZE, EMBED_SIZE, x2);

  // Store K and V into the cache
  memcpy(kv_cache_k[layer_i][pos], qkv + EMBED_SIZE, EMBED_SIZE * sizeof(float));
  memcpy(kv_cache_v[layer_i][pos], qkv + 2 * EMBED_SIZE, EMBED_SIZE * sizeof(float));

  // ── 3. Multi-head attention ───────────────────────────────────────────────
  memset(attn_out, 0, sizeof(attn_out));

  int hs = HEAD_SIZE;  // EMBED_SIZE / NUM_HEADS

  for (int h = 0; h < NUM_HEADS; h++) {
    const float* q_h = qkv + h * hs;

    // Compute attention scores against all cached positions 0..pos
    float scores[BLOCK_SIZE];
    float scale = 1.0f / sqrtf((float)hs);
    for (int t = 0; t <= pos; t++) {
      const float* k_t = kv_cache_k[layer_i][t] + h * hs;
      float dot = 0;
      for (int d = 0; d < hs; d++) dot += q_h[d] * k_t[d];
      scores[t] = dot * scale;
    }

    // Softmax over valid positions
    float maxs = scores[0];
    for (int t = 1; t <= pos; t++)
      if (scores[t] > maxs) maxs = scores[t];
    float sum = 0;
    for (int t = 0; t <= pos; t++) {
      scores[t] = expf(scores[t] - maxs);
      sum += scores[t];
    }
    for (int t = 0; t <= pos; t++) scores[t] /= sum;

    // Weighted sum of V
    for (int t = 0; t <= pos; t++) {
      const float* v_t = kv_cache_v[layer_i][t] + h * hs;
      for (int d = 0; d < hs; d++)
        attn_out[h * hs + d] += scores[t] * v_t[d];
    }
  }

  // ── 4. Attention output projection ───────────────────────────────────────
  matvec(x2, W.attn_proj_w, nullptr, EMBED_SIZE, EMBED_SIZE, attn_out);

  // ── 5. Residual ───────────────────────────────────────────────────────────
  for (int i = 0; i < EMBED_SIZE; i++) vec[i] += x2[i];

  // ── 6. LayerNorm 2 ────────────────────────────────────────────────────────
  layer_norm(x2, vec, W.ln2_w, W.ln2_b, EMBED_SIZE);

  // ── 7. MLP: fc -> GELU -> proj ────────────────────────────────────────────
  matvec(mlp_hidden, W.mlp_fc_w, W.mlp_fc_b, 4 * EMBED_SIZE, EMBED_SIZE, x2);
  for (int i = 0; i < 4 * EMBED_SIZE; i++) mlp_hidden[i] = gelu(mlp_hidden[i]);
  matvec(x2, W.mlp_proj_w, W.mlp_proj_b, EMBED_SIZE, 4 * EMBED_SIZE, mlp_hidden);

  // ── 8. Residual ───────────────────────────────────────────────────────────
  for (int i = 0; i < EMBED_SIZE; i++) vec[i] += x2[i];
}

// ─── Full forward pass for ONE new token ─────────────────────────────────────
// Returns the next token index.
int forward_and_sample(int new_token, int pos) {
  // Token + position embedding (from PROGMEM)
  for (int i = 0; i < EMBED_SIZE; i++) {
    x[i] = pgm_float(tok_emb + (size_t)new_token * EMBED_SIZE + i)
           + pgm_float(pos_emb + (size_t)pos * EMBED_SIZE + i);
  }

  // Transformer blocks
  for (int l = 0; l < NUM_LAYERS; l++)
    transformer_block(l, pos, x);

  // Final layer norm
  layer_norm(x2, x, ln_f_w, ln_f_b, EMBED_SIZE);

  // LM head: [VOCAB_SIZE x EMBED_SIZE] * x2 -> logits
  matvec(logits, lm_head_w, nullptr, VOCAB_SIZE, EMBED_SIZE, x2);

  // ── Temperature ──────────────────────────────────────────────────────────
  if (TEMPERATURE != 1.0f)
    for (int i = 0; i < VOCAB_SIZE; i++) logits[i] /= TEMPERATURE;

  // ── Top-K masking ────────────────────────────────────────────────────────
  if (TOP_K > 0 && TOP_K < VOCAB_SIZE) {
    // Find the K-th largest value
    float tmp[VOCAB_SIZE];
    memcpy(tmp, logits, sizeof(logits));
    // Partial selection sort to find threshold
    float kth = -1e30f;
    for (int k = 0; k < TOP_K; k++) {
      float best = -1e30f;
      int bi = 0;
      for (int i = 0; i < VOCAB_SIZE; i++)
        if (tmp[i] > best) {
          best = tmp[i];
          bi = i;
        }
      kth = best;
      tmp[bi] = -1e30f;
    }
    for (int i = 0; i < VOCAB_SIZE; i++)
      if (logits[i] < kth) logits[i] = -1e30f;
  }

  softmax(logits, VOCAB_SIZE);

  // ── Sample ────────────────────────────────────────────────────────────────
  float r = (float)esp_random() / (float)UINT32_MAX;
  float cum = 0;
  for (int i = 0; i < VOCAB_SIZE; i++) {
    cum += logits[i];
    if (r < cum) return i;
  }
  return VOCAB_SIZE - 1;
}

// ─── LCD helpers ─────────────────────────────────────────────────────────────
static char lcd_buf[2][17];  // two rows, 16 chars + null
static int lcd_col = 0, lcd_row = 0;

void lcd_clear_print() {
  lcd.clear();
  lcd_col = 0;
  lcd_row = 0;
  memset(lcd_buf, ' ', sizeof(lcd_buf));
  lcd_buf[0][16] = lcd_buf[1][16] = 0;
}

void lcd_push_char(char c) {
  if (c == '\n') {
    // Scroll: row 1 -> row 0, clear row 1
    memcpy(lcd_buf[0], lcd_buf[1], 16);
    memset(lcd_buf[1], ' ', 16);
    lcd_col = 0;
    lcd_row = 1;
  } else {
    if (lcd_col >= 16) {
      // Wrap: scroll up
      memcpy(lcd_buf[0], lcd_buf[1], 16);
      memset(lcd_buf[1], ' ', 16);
      lcd_col = 0;
      lcd_row = 1;
    }
    lcd_buf[lcd_row][lcd_col++] = c;
  }
  // Redraw both rows
  lcd.setCursor(0, 0);
  lcd.print(lcd_buf[0]);
  lcd.setCursor(0, 1);
  lcd.print(lcd_buf[1]);
}

// ─── Encode a string into token_buf ──────────────────────────────────────────
bool encode_prompt(const String& s) {
  ctx_len = 0;
  for (int i = 0; i < (int)s.length() && ctx_len < BLOCK_SIZE; i++) {
    char c = s[i];
    if ((uint8_t)c >= 128) continue;  // skip non-ASCII
    int16_t idx = (int16_t)pgm_read_word(STOI + (uint8_t)c);
    if (idx < 0) {
      Serial.print("Warning: unknown char '");
      Serial.print(c);
      Serial.println("' skipped");
      continue;
    }
    token_buf[ctx_len++] = idx;
  }
  return ctx_len > 0;
}

// ─── SETUP ───────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  // LCD init
  Wire.begin(21, 22);  // SDA, SCL
  lcd.init();
  lcd.backlight();
  lcd_clear_print();
  lcd.setCursor(0, 0);
  lcd.print("ESP32 MicroLLM");
  lcd.setCursor(0, 1);
  lcd.print("Booting...");

  // Wire layer weight pointers — must match weights.h variable names
  layer_wts[0] = {
    b0_ln1_w, b0_ln1_b,
    b0_ln2_w, b0_ln2_b,
    b0_attn_qkv_w,
    b0_attn_proj_w,
    b0_mlp_fc_w, b0_mlp_fc_b,
    b0_mlp_proj_w, b0_mlp_proj_b
  };
  layer_wts[1] = {
    b1_ln1_w, b1_ln1_b,
    b1_ln2_w, b1_ln2_b,
    b1_attn_qkv_w,
    b1_attn_proj_w,
    b1_mlp_fc_w, b1_mlp_fc_b,
    b1_mlp_proj_w, b1_mlp_proj_b
  };

  Serial.println("=================================");
  Serial.println(" ESP32 MicroLLM — Ready!");
  Serial.printf(" Vocab: %d | Embed: %d | Layers: %d\n",
                VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS);
  Serial.println(" Type a prompt and press Enter.");
  Serial.println("=================================");

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Ready! Type on");
  lcd.setCursor(0, 1);
  lcd.print("Serial Monitor");
}

// ─── LOOP ────────────────────────────────────────────────────────────────────
void loop() {
    // Flush any garbage that accumulated during generation
    while (Serial.available()) Serial.read();
    
    Serial.println("Ready. Type your prompt:");
    
    // Wait properly for user input — don't proceed until Enter is pressed
    String prompt = "";
    bool got_input = false;
    
    while (!got_input) {
        if (Serial.available()) {
            char c = Serial.read();
            
            if (c == '\n' || c == '\r') {
                if (prompt.length() > 0) {
                    got_input = true;  // valid input received
                }
                // else ignore empty enter presses
            } else if (c >= 32 && c < 127) {
                prompt += c;  // only add printable ASCII
            }
            // anything else (garbage, ? symbols) is silently ignored
        }
    }
    
    // One final flush before we start generating
    delay(10);
    while (Serial.available()) Serial.read();

    prompt.trim();

    Serial.println("----------------------------------------");
    Serial.print("Prompt: "); Serial.println(prompt);
    Serial.print("Output: ");

    lcd_clear_print();
    for (int i = 0; i < (int)prompt.length() && i < 32; i++)
        lcd_push_char(prompt[i]);
    delay(800);
    lcd_clear_print();

    if (!encode_prompt(prompt)) {
        Serial.println("[ERROR] Could not encode prompt — unknown characters.");
        return;
    }

    // Clear KV cache
    memset(kv_cache_k, 0, sizeof(kv_cache_k));
    memset(kv_cache_v, 0, sizeof(kv_cache_v));

    // Feed prompt tokens into KV cache
    int last_token = token_buf[0];
    for (int p = 0; p < ctx_len; p++) {
        last_token = token_buf[p];
        if (p < ctx_len - 1) {
            for (int i = 0; i < EMBED_SIZE; i++) {
                x[i] = pgm_float(tok_emb + (size_t)last_token * EMBED_SIZE + i)
                      + pgm_float(pos_emb + (size_t)p         * EMBED_SIZE + i);
            }
            for (int l = 0; l < NUM_LAYERS; l++)
                transformer_block(l, p, x);
        }
    }

    // Generate new tokens
    int pos = ctx_len - 1;
    int cur_token = last_token;

    for (int gen = 0; gen < MAX_NEW_TOKENS; gen++) {
        int next_token = forward_and_sample(cur_token, pos);
        pos++;

        char ch = (char)pgm_read_byte(ITOS + next_token);
        Serial.print(ch);

        if (ch == '\n' || (ch >= 32 && ch < 127))
            lcd_push_char(ch);

        cur_token = next_token;

        if (pos >= BLOCK_SIZE - 1) {
            Serial.print(" [CTX FULL]");
            break;
        }

        if (gen % 10 == 0) yield();
    }

    Serial.println();
    Serial.println("----------------------------------------");
}