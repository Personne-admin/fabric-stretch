#include <Arduino.h>
#include <EEPROM.h>
#include <OneWire.h>
#include <SPI.h>
#include <U8x8lib.h>
#include <Wire.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <digitalWriteFast.h>

#include "Camera.hh"
#include "Commander.hh"
#include "Constants.hh"
#include "Display.hh"
#include "Light.hh"
#include "Stepper.hh"
#include "Tensiometer.hh"

namespace CurrentSensor {

constexpr uint8_t kAddr = 0x40;
constexpr uint16_t kConfig = 0x1557;
constexpr uint16_t kCalib = 8192;
constexpr float kDivisor = 20000.0f;

void init() {
  Wire.begin();
  Wire.beginTransmission(kAddr);
  Wire.write(0x00);
  Wire.write(kConfig >> 8);
  Wire.write(kConfig & 0xFF);
  Wire.endTransmission();

  Wire.beginTransmission(kAddr);
  Wire.write(0x05);
  Wire.write(kCalib >> 8);
  Wire.write(kCalib & 0xFF);
  Wire.endTransmission();
}

float read_current() {
  Wire.beginTransmission(kAddr);
  Wire.write(0x05);
  Wire.write(kCalib >> 8);
  Wire.write(kCalib & 0xFF);
  Wire.endTransmission();

  Wire.beginTransmission(kAddr);
  Wire.write(0x04);
  Wire.endTransmission();

  Wire.requestFrom(kAddr, (uint8_t)2);
  int16_t v = (Wire.read() << 8) | Wire.read();
  return v / kDivisor;
}

}  // namespace CurrentSensor

ISR(TIMER2_COMPA_vect) {
  Stepper::isr_callback();
  Tension::isr_callback(digitalReadFast(Constants::Pin::SpiClk),
                        digitalReadFast(Constants::Pin::SpiData));
  Stepper::check_force_limit_from_isr();
}

void setup() {
  Serial.begin(Constants::kSerialSpeed);
  Serial.setTimeout(Constants::kSerialTimeout);

  pinMode(Constants::Pin::MotorEnable, OUTPUT);
  pinMode(Constants::Pin::MotorDir, OUTPUT);
  pinMode(Constants::Pin::MotorStep, OUTPUT);

  pinMode(Constants::Pin::RedLED, OUTPUT);
  pinMode(Constants::Pin::GreenLED, OUTPUT);
  pinMode(Constants::Pin::BlueLED, OUTPUT);

  pinMode(Constants::Pin::SpiClk, INPUT_PULLUP);
  pinMode(Constants::Pin::SpiData, INPUT_PULLUP);

  pinMode(Constants::Pin::Camera, OUTPUT);
  pinMode(Constants::Pin::DigPotCS, OUTPUT);
  pinMode(Constants::Pin::OneWire, OUTPUT);

  pinMode(Constants::Pin::IrqFirst, INPUT_PULLUP);
  pinMode(Constants::Pin::IrqSecond, INPUT_PULLUP);

  digitalWriteFast(Constants::Pin::RedLED, HIGH);
  digitalWriteFast(Constants::Pin::GreenLED, HIGH);
  digitalWriteFast(Constants::Pin::BlueLED, HIGH);

  while (!Serial) {
  }
  Stepper::init();
  Tension::load_config();
  CurrentSensor::init();
  SPI.begin();
  Lights::off();
  Lights::search_and_print();
  Display::init();

  EIFR = bit(INTF1) | bit(INTF0);
  attachInterrupt(
    digitalPinToInterrupt(Constants::Pin::IrqSecond),
    []() {
      if (Stepper::is_busy())
        Stepper::stop_from_isr();
      else {
        digitalWriteFast(Constants::Pin::GreenLED, HIGH);
        Stepper::move_from_isr(-400);
      }
    },
    FALLING);

  attachInterrupt(
    digitalPinToInterrupt(Constants::Pin::IrqFirst),
    []() {
      if (Stepper::is_busy())
        Stepper::stop_from_isr();
      else {
        digitalWriteFast(Constants::Pin::RedLED, HIGH);
        Stepper::move_from_isr(400);
      }
    },
    FALLING);

  digitalWriteFast(Constants::Pin::RedLED, LOW);
  digitalWriteFast(Constants::Pin::GreenLED, LOW);
  digitalWriteFast(Constants::Pin::BlueLED, LOW);

  Serial.println(F("\nstretch-motor ready"));
}

void loop() {
  Command::poll();
  Stepper::poll();
  Display::update();
  Camera::update();
}
