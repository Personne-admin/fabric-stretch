#ifndef STEPPER_HH
#define STEPPER_HH

#include <avr/interrupt.h>
#include <digitalWriteFast.h>
#include <stdint.h>

#include "Constants.hh"
#include "Tensiometer.hh"

namespace Stepper {

enum class AbortReason : uint8_t { None, Host, Panic, ForceLimit, Replaced };

namespace Detail {

volatile bool running{ false };
volatile uint32_t remaining{ 0 };
volatile uint8_t fraction{ 0 };
volatile bool move_complete{ false };
volatile AbortReason aborted{ AbortReason::None };
volatile AbortReason previous_aborted{ AbortReason::None };
volatile uint32_t previous_aborted_remaining{ 0 };
volatile uint32_t aborted_remaining{ 0 };

volatile bool limit_abort{ false };

}  // namespace Detail

constexpr uint8_t kMinFrameBits = 24;
constexpr uint8_t kTripConfirmFrames = 3;

void set_speed_multiplier(uint8_t mult) {
  if (mult < 1)
    mult = 1;

  uint16_t ticks =
    static_cast<uint16_t>(Constants::Timer::BaseOcr2a + 1) / mult;
  if (ticks < 2)
    ticks = 2;

  uint8_t new_ocr = static_cast<uint8_t>(ticks - 1);

  noInterrupts();
  OCR2A = new_ocr;
  interrupts();
}

void init() {
  noInterrupts();
  TCCR2A = 0;
  TCCR2B = 0;
  TCNT2 = 0;
  TIMSK2 = 0;
  TCCR2A = (1 << WGM21);
  TCCR2B = (1 << CS21);
  OCR2A = Constants::Timer::BaseOcr2a;
  TIMSK2 = (1 << OCIE2A);
  interrupts();

  set_speed_multiplier(1);
}

void enable(bool on, bool direction_positive = true) {
  digitalWriteFast(Constants::Pin::MotorEnable, on ? LOW : HIGH);
  digitalWriteFast(Constants::Pin::MotorDir, direction_positive ? HIGH : LOW);
}

void stop_from_isr(AbortReason reason = AbortReason::Panic) {
  if (reason == AbortReason::Panic) {
    Tension::Detail::limit_active = false;
    Tension::Detail::tripped = false;
  }
  if (Detail::running) {
    // A force trip can start auto-release before the main loop reports the first abort.
    if (Detail::aborted != AbortReason::None) {
      Detail::previous_aborted = Detail::aborted;
      Detail::previous_aborted_remaining = Detail::aborted_remaining;
    }
    Detail::aborted = reason;
    Detail::aborted_remaining = Detail::remaining;
  }
  Detail::running = false;
  Detail::remaining = 0;
  Detail::move_complete = false;
  digitalWriteFast(Constants::Pin::MotorStep, LOW);
  digitalWriteFast(Constants::Pin::MotorEnable, HIGH);
  digitalWriteFast(Constants::Pin::RedLED, LOW);
  digitalWriteFast(Constants::Pin::GreenLED, LOW);
}

uint32_t stop() {
  const uint8_t saved = SREG;
  noInterrupts();
  const uint32_t remaining = Detail::remaining;
  // Disarm in the same critical section so a trip cannot restart an auto-release.
  Tension::Detail::limit_active = false;
  Tension::Detail::tripped = false;
  stop_from_isr(AbortReason::Host);
  SREG = saved;
  return remaining;
}

void move_from_isr(int32_t steps) {
  if (steps == 0)
    return;

  if (Detail::running)
    stop_from_isr(AbortReason::Replaced);
  digitalWriteFast(Constants::Pin::MotorEnable, LOW);
  digitalWriteFast(Constants::Pin::MotorDir, steps > 0 ? HIGH : LOW);
  Detail::remaining = steps > 0 ? static_cast<uint32_t>(steps) : -static_cast<uint32_t>(steps);
  Detail::fraction = Constants::Timer::Fraction;
  digitalWriteFast(Constants::Pin::MotorStep, LOW);
  Detail::move_complete = false;
  Detail::running = true;
}

void move(int32_t steps) {
  const uint8_t saved = SREG;
  noInterrupts();
  move_from_isr(steps);
  SREG = saved;
}

bool is_busy() {
  return Detail::running;
}

uint32_t steps_remaining() {
  return Detail::remaining;
}

void check_force_limit_from_isr() {
  static uint8_t over_count = 0;

  if (!Tension::Detail::limit_active) {
    over_count = 0;
    return;
  }
  if (!Tension::Detail::unread_isr_data)
    return;

  Tension::Detail::unread_isr_data = false;

  if (Tension::Detail::last_frame_bits < kMinFrameBits)
    return;

  const int16_t raw_counts = Tension::raw_counts();
  const float force_n = Tension::force_newtons_from_raw(raw_counts);

  if (fabs(force_n) < Tension::Detail::limit_force_n) {
    over_count = 0;
    return;
  }
  if (++over_count < kTripConfirmFrames)
    return;
  over_count = 0;

  Tension::Detail::tripped = true;
  Tension::Detail::tripped_force = force_n;
  Tension::Detail::tripped_raw_counts = raw_counts;
  Tension::Detail::limit_active = false;
  Detail::limit_abort = true;

  stop_from_isr(AbortReason::ForceLimit);
  if (Tension::Detail::auto_release_steps > 0) {
    int32_t rel = Tension::Detail::auto_release_steps;
    move_from_isr(Tension::Detail::release_dir_positive ? rel : -rel);
  }
}

void isr_callback() {
  if (Detail::running) {
    digitalToggleFast(Constants::Pin::MotorStep);
    if (--Detail::fraction == 0) {
      if (Detail::remaining > 0) {
        Detail::remaining--;
        Detail::fraction = Constants::Timer::Fraction;
      } else {
        Detail::running = false;
        Detail::move_complete = true;
        digitalWriteFast(Constants::Pin::MotorEnable, HIGH);
        digitalWriteFast(Constants::Pin::RedLED, LOW);
        digitalWriteFast(Constants::Pin::GreenLED, LOW);
      }
    }
  }
}

void print_abort(AbortReason aborted, uint32_t remaining) {
  if (aborted == AbortReason::None)
    return;
  Serial.print(F("MOVE_ABORTED reason="));
  switch (aborted) {
    case AbortReason::Host: Serial.print(F("host_stop")); break;
    case AbortReason::Panic: Serial.print(F("panic")); break;
    case AbortReason::ForceLimit: Serial.print(F("force_limit")); break;
    case AbortReason::Replaced: Serial.print(F("replaced")); break;
    default: break;
  }
  Serial.print(F(" remaining="));
  Serial.println(remaining);
}

void poll() {
  const uint8_t saved = SREG;
  noInterrupts();
  const AbortReason aborted = Detail::aborted;
  const AbortReason previous_aborted = Detail::previous_aborted;
  const uint32_t previous_remaining = Detail::previous_aborted_remaining;
  const uint32_t remaining = Detail::aborted_remaining;
  const bool limit_abort = Detail::limit_abort;
  const bool complete = Detail::move_complete;
  Detail::aborted = AbortReason::None;
  Detail::previous_aborted = AbortReason::None;
  Detail::limit_abort = false;
  Detail::move_complete = false;
  SREG = saved;
  print_abort(previous_aborted, previous_remaining);
  print_abort(aborted, remaining);
  if (limit_abort) {
    Serial.print(F("LIMIT_TRIP force_n="));
    Serial.print(Tension::tripped_force(), 6);
    Serial.print(F(" raw_counts="));
    Serial.print(Tension::Detail::tripped_raw_counts);
    Serial.print(F(" zero_counts="));
    Serial.print(Tension::Detail::zero);
    Serial.print(F(" calibration_n_per_count="));
    Serial.print(Tension::Detail::coef, 8);
    Serial.print(F(" limit_n="));
    Serial.println(Tension::Detail::limit_force_n, 6);
  }
  if (complete) {
    Serial.println(F("MOVE_DONE"));
  }
}

}  // namespace Stepper

#endif /* STEPPER_HH */
