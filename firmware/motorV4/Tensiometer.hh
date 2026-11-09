#ifndef TENSIOMETER_HH
#define TENSIOMETER_HH

#include <stdint.h>
#include <math.h>

namespace Tension {

constexpr int kSpiMax = 48;
constexpr int kSpiCount = 5;

constexpr int EepromZeroAddr = 80;
constexpr int EepromCoefAddr = 84;

namespace Detail {

volatile uint8_t buffer[kSpiMax]{};
volatile uint8_t safe_buffer[kSpiMax]{};
volatile uint8_t pointer{ 0 };
volatile uint8_t count{ kSpiCount };
volatile bool unread_isr_data{ false };
volatile uint8_t last_frame_bits{ 0 };

volatile uint8_t ready{ 0 };

int16_t zero{ 0 };
float coef{ 0.0125f };

union SpiValue {
  int32_t i32;
  int16_t i16[2];
  uint8_t u8[4];
};

SpiValue value{ 0 };

volatile bool limit_active{ false };
volatile float limit_force_n{ 0.0f };
volatile bool tripped{ false };
volatile float tripped_force{ 0.0f };
volatile int16_t tripped_raw_counts{ 0 };
int32_t auto_release_steps{ 0 };
bool release_dir_positive{ false };

};  // namespace Detail

void isr_callback(bool clk_high, bool data_high) {
  if (clk_high)
    Detail::count = 0;
  else {
    if (Detail::count == 0) {
      Detail::count++;
      if (Detail::pointer < kSpiMax)
        Detail::buffer[Detail::pointer++] = data_high;
    } else if (Detail::count < kSpiCount) {
      Detail::count++;
      if (Detail::count == kSpiCount) {
        for(uint8_t i = 0; i < Detail::pointer; i++) {
            Detail::safe_buffer[i] = Detail::buffer[i];
        }
        Detail::ready = Detail::pointer;
        Detail::last_frame_bits = Detail::pointer;
        Detail::unread_isr_data = true;
        Detail::pointer = 0;
      }
    }
  }
}

bool is_ready() {
  return Detail::ready >= 24;
}

uint8_t get_ready_count() {
  return Detail::ready;
}

void clear_ready() {
  Detail::ready = 0;
}

int16_t raw_counts() {
  for (uint8_t i = 0, k = 3; k > 1; k--)
    for (uint8_t j = 128; j != 0; j >>= 1, i++)
      if (Detail::safe_buffer[i])
        Detail::value.u8[k] |= j;
      else
        Detail::value.u8[k] &= ~j;
  return Detail::value.i16[1];
}

// This is the only raw-count to force conversion.  Both the `t` telemetry
// command and the in-motion overload check call it, so N is never compared
// with ADC counts or a second, differently-scaled value.
float force_newtons_from_raw(int16_t raw) {
  return Detail::coef * static_cast<float>(Detail::zero - raw);
}

float convert() {
  return force_newtons_from_raw(raw_counts());
}

void load_config() {
  EEPROM.get(EepromZeroAddr, Detail::zero);
  EEPROM.get(EepromCoefAddr, Detail::coef);
}

void tare() {
  Detail::zero = Detail::value.i16[1];
  EEPROM.put(EepromZeroAddr, Detail::zero);
}

void set_multiplier(float c) {
  Detail::coef = c;
  EEPROM.put(EepromCoefAddr, Detail::coef);
}


void set_force_limit(float force_n, int32_t release_steps,
                     bool release_dir_positive) {
  if (force_n <= 0.0f || Detail::coef == 0.0f) {
    noInterrupts();
    Detail::limit_active = false;
    interrupts();
    return;
  }

  noInterrupts();
  Detail::limit_force_n = force_n;
  Detail::auto_release_steps = release_steps;
  Detail::release_dir_positive = release_dir_positive;
  Detail::tripped = false;
  Detail::limit_active = true;
  interrupts();
}

void disarm_force_limit() {
  noInterrupts();
  Detail::limit_active = false;
  Detail::tripped = false;
  interrupts();
}

bool was_tripped() {
  return Detail::tripped;
}

void clear_tripped() {
  noInterrupts();
  Detail::tripped = false;
  interrupts();
}

float tripped_force() {
  return Detail::tripped_force;
}

};  // namespace Tension

#endif /* TENSIOMETER_HH */
