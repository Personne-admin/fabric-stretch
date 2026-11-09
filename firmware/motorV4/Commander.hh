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

// Serial command parser. The command and reply table is documented in firmware/README.md;
// the host side is fabsim.rig.hardware.protocol.
namespace Command {

// Handles at most one command per call (called from loop()).
void poll() {
  if (Serial.available() == 0)
    return;

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

    case 't':
      {
        uint32_t motor_t0 = millis();
        while (Stepper::is_busy() && (millis() - motor_t0) < 30000)
          Stepper::poll();

        if (Stepper::is_busy()) {
          Serial.println(F("tension: N/A"));
          break;
        }

        uint32_t t0 = millis();
        while ((millis() - t0) < 300)
          if (Tension::is_ready())
            break;

        if (Tension::is_ready()) {
          noInterrupts();
          int16_t raw_counts = Tension::raw_counts();
          float v = Tension::force_newtons_from_raw(raw_counts);
          interrupts();
          Tension::clear_ready();
          Serial.print(F("TENSION force_n="));
          Serial.print(v, 6);
          Serial.print(F(" raw_counts="));
          Serial.print(raw_counts);
          Serial.print(F(" zero_counts="));
          Serial.print(Tension::Detail::zero);
          Serial.print(F(" calibration_n_per_count="));
          Serial.println(Tension::Detail::coef, 8);
        } else {
          Serial.println(F("tension: N/A"));
        }
      }
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
}

}  // namespace Command

#endif /* COMMANDER_HH */
