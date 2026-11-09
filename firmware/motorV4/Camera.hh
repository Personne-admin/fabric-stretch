#ifndef CAMERA_HH
#define CAMERA_HH

#include <Arduino.h>
#include <digitalWriteFast.h>
#include <stdint.h>

#include "Constants.hh"

namespace Camera {

namespace Detail {

bool active{false};
uint32_t start_time{0};
uint16_t pulse_duration{20};

} // namespace Detail

void trigger(uint16_t pulse_ms = 20) {
  Detail::pulse_duration = pulse_ms;
  digitalWriteFast(Constants::Pin::Camera, HIGH);
  Detail::start_time = millis();
  Detail::active = true;
}

void update() {
  if (!Detail::active)
    return;

  if ((millis() - Detail::start_time) >= Detail::pulse_duration) {
    digitalWriteFast(Constants::Pin::Camera, LOW);
    Detail::active = false;
  }
}

bool is_busy() { return Detail::active; }

} // namespace Camera

#endif /* CAMERA_HH */
