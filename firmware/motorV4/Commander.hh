#ifndef COMMANDER_HH
#define COMMANDER_HH

#include <Arduino.h>
#include <digitalWriteFast.h>
#include <stdint.h>

#include "Camera.hh"
#include "Constants.hh"
#include "Light.hh"
#include "Stepper.hh"
#include "Tensiometer.hh"

// Serial command parser. Commands and replies are documented in docs/serial_protocol.md;
// the host side is fabsim.rig.hardware.protocol.
namespace Command {

bool tension_pending = false;
uint32_t tension_requested_at = 0;
uint32_t tension_idle_at = 0;
bool tension_waiting_motor = false;

void poll_tension() {
  if (!tension_pending)
    return;
  const uint32_t now = millis();
  if (Stepper::is_busy()) {
    if (now - tension_requested_at >= 30000) {
      tension_pending = false;
      Serial.println(F("tension: N/A"));
    }
    return;
  }
  if (tension_waiting_motor) {
    tension_waiting_motor = false;
    tension_idle_at = now;
  }
  if (!Tension::is_ready()) {
    if (now - tension_idle_at >= 300) {
      tension_pending = false;
      Serial.println(F("tension: N/A"));
    }
    return;
  }
  const uint8_t saved = SREG;
  noInterrupts();
  const int16_t raw_counts = Tension::raw_counts();
  const int16_t zero = Tension::Detail::zero;
  const float coef = Tension::Detail::coef;
  Tension::clear_ready();
  SREG = saved;
  const float force = coef * (static_cast<float>(zero) - raw_counts);
  tension_pending = false;
  Serial.print(F("TENSION force_n="));
  Serial.print(force, 6);
  Serial.print(F(" raw_counts="));
  Serial.print(raw_counts);
  Serial.print(F(" zero_counts="));
  Serial.print(zero);
  Serial.print(F(" calibration_n_per_count="));
  Serial.println(coef, 8);
}

// Handles at most one command per call (called from loop()).
void poll() {
  if (Serial.available() == 0) {
    poll_tension();
    return;
  }

  int c = Serial.read();

  switch (c) {

    case '+':
      {
        int32_t n = Serial.parseInt();
        digitalWriteFast(Constants::Pin::RedLED, HIGH);
        Stepper::move(n);
        Serial.print(F("jog: +"));
        Serial.println(n);
      }
      break;

    case '-':
      {
        int32_t n = Serial.parseInt();
        digitalWriteFast(Constants::Pin::GreenLED, HIGH);
        Stepper::move(-n);
        Serial.print(F("jog: -"));
        Serial.println(n);
      }
      break;

    case 'L':
      {
        uint8_t m = static_cast<uint8_t>(Serial.parseInt());
        Lights::set_mask(m);
        if (m != 0) {
          Serial.print(F("leds: "));
          Serial.print(m);
          Serial.println(F(" OK"));
        } else {
          Serial.print(F("leds: "));
          Serial.println(m);
        }
      }
      break;

    case 'P':
      {
        uint8_t p = static_cast<uint8_t>(Serial.parseInt());
        Lights::set_power(p);
        Serial.print(F("power: "));
        Serial.println(p);
      }
      break;

    case 'V':
      Stepper::set_speed_multiplier(static_cast<uint8_t>(Serial.parseInt()));
      break;

    case '#':
      Camera::trigger(20);
      Serial.println(F("trigger"));
      break;

    case '@':
      Tension::tare();
      Serial.print(F("zero: "));
      Serial.println(Tension::Detail::zero);
      break;

    case '*':
      {
        float k = Serial.parseFloat();
        if (k != 0.0f) {
          Tension::set_multiplier(k);
          Serial.print(F("coef: "));
          Serial.println(k, 6);
        }
      }
      break;

    case 'S':
      {
        const uint32_t remaining = Stepper::stop();
        if (tension_pending) {
          tension_pending = false;
          Serial.println(F("tension: N/A"));
        }
        Stepper::poll();
        Serial.print(F("STOPPED remaining="));
        Serial.println(remaining);
      }
      break;

    case 't':
      tension_pending = true;
      tension_requested_at = millis();
      tension_idle_at = tension_requested_at;
      tension_waiting_motor = Stepper::is_busy();
      break;

    case 'i':
      Serial.println(F("status: API ready"));
      break;

    case 'F':
      {
        float f = Serial.parseFloat();

        int32_t rel = 0;
        int dir = 0;
        if (Serial.peek() == ',') {
          Serial.read();
          rel = Serial.parseInt();
          if (Serial.peek() == ',') {
            Serial.read();
            dir = Serial.parseInt();
          }
        }
        if (f <= 0.0f) {
          Tension::disarm_force_limit();
          Serial.println(F("limit: off"));
        } else {
          Tension::set_force_limit(f, rel, dir != 0);
          Serial.print(F("LIMIT_ARMED limit_n="));
          Serial.print(f, 6);
          Serial.print(F(" release_motor_steps="));
          Serial.print(rel);
          Serial.print(F(" release_dir_positive="));
          Serial.println(dir);
        }
      }
      break;

    case ' ':
    case '\r':
    case '\n':
    case '\t':
      break;

    default:
      Serial.print('?');
      Serial.println(static_cast<char>(c));
      break;
  }
  poll_tension();
}

}  // namespace Command

#endif /* COMMANDER_HH */
