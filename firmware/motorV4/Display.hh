#ifndef DISPLAY_HH
#define DISPLAY_HH

#include "Light.hh"

U8X8_SSD1306_128X64_NONAME_HW_I2C u8x8(U8X8_PIN_NONE);

namespace Display {

uint32_t tension_clear_time{0};

void init() {
  u8x8.begin();
  u8x8.setFlipMode(true);
  u8x8.setFont(u8x8_font_7x14B_1x2_r);
  u8x8.setCursor(0, 6);
  u8x8.print(F(" stretch-motor 2"));
}

void update() {
  if (Tension::is_ready()) {
    float t = Tension::convert();
    u8x8.setCursor(0, 0);
    u8x8.print(F("tense: "));
    u8x8.print(t, 1);
    u8x8.print(F("    "));
    tension_clear_time = millis() + 100;
  }
  if (tension_clear_time && millis() > tension_clear_time) {
    tension_clear_time = 0;
    u8x8.clearLine(0);
    u8x8.clearLine(1);
  }

  if (Stepper::steps_remaining()) {
    u8x8.setCursor(0, 2);
    u8x8.print(F("steps: "));
    u8x8.print(Stepper::steps_remaining());
    u8x8.print(F("    "));
  } else {
    u8x8.clearLine(2);
    u8x8.clearLine(3);
  }

  if (Lights::mask()) {
    u8x8.setCursor(0, 4);
    u8x8.print(F("leds: "));
    if (Lights::mask() < 16)
      u8x8.print('0');
    u8x8.print(Lights::mask(), HEX);
    u8x8.print(F(" / "));
    u8x8.print(Lights::power());
    u8x8.print(F("  "));
  } else {
    u8x8.clearLine(4);
    u8x8.clearLine(5);
  }
}

} // namespace Display

#endif /* DISPLAY_HH */
