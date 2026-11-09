#ifndef LIGHT_HH
#define LIGHT_HH

#include <OneWire.h>
#include <SPI.h>
#include <avr/pgmspace.h>
#include <digitalWriteFast.h>
#include <stdint.h>

#include "Constants.hh"

namespace Lights {

namespace Detail {

constexpr uint8_t kDeviceCount = 4;
constexpr uint8_t kFamilyId = 0x3A;
constexpr uint8_t kAccessWrite = 0x5A;
constexpr uint8_t kAckSuccess = 0xAA;
constexpr uint8_t kPotWriteCmd = 0x11;

const char addresses[kDeviceCount][8] PROGMEM = {
    {0x3A, 0x17, 0x95, 0x62, 0x00, 0x00, 0x00, 0xB3},
    {0x3A, 0xA6, 0x95, 0x62, 0x00, 0x00, 0x00, 0x83},
    {0x3A, 0xF6, 0x89, 0x62, 0x00, 0x00, 0x00, 0xF0},
    {0x3A, 0xEE, 0x92, 0x62, 0x00, 0x00, 0x00, 0x06}};

OneWire bus(Constants::Pin::OneWire);

uint8_t current_mask{0};
uint8_t current_power{0};

void load_address(uint8_t index, uint8_t addr[8]) {
  for (uint8_t j = 0; j < 8; j++)
    addr[j] = pgm_read_byte(&addresses[index][j]);
}

bool write_gpio(uint8_t index, uint8_t state) {
  uint8_t addr[8];
  load_address(index, addr);
  state &= 0x03;
  bus.reset();
  bus.select(addr);
  bus.write(kAccessWrite);
  bus.write(~state);
  bus.write(state);
  uint8_t ack = bus.read();

  if (ack == kAckSuccess)
    bus.read();
  bus.reset();
  return ack == kAckSuccess;
}

} // namespace Detail

uint8_t mask() { return Detail::current_mask; }

uint8_t power() { return Detail::current_power; }

void set_mask(uint8_t mask) {
  Detail::current_mask = mask;
  uint8_t m = mask;
  for (uint8_t i = 0; i < Detail::kDeviceCount; i++) {
    Detail::write_gpio(i, m & 0x03);
    m >>= 2;
  }
}

void set_power(uint8_t power) {
  Detail::current_power = power;
  digitalWriteFast(Constants::Pin::DigPotCS, LOW);
  SPI.transfer(Detail::kPotWriteCmd);
  SPI.transfer(255 - power);
  digitalWriteFast(Constants::Pin::DigPotCS, HIGH);
}

void off() {
  set_mask(0);
  set_power(0);
}

void search_and_print() {
  uint8_t addr[8];
  Serial.println(F("\nOneWire search:"));
  Detail::bus.reset();
  Detail::bus.reset_search();
  while (Detail::bus.search(addr)) {
    for (uint8_t i = 0; i < 8; i++) {
      if (i)
        Serial.print(':');
      if (addr[i] < 0x10)
        Serial.print('0');

      Serial.print(addr[i], HEX);
    }

    Serial.println();
  }
}

} // namespace Lights

#endif /* LIGHT_HH */
