import struct
import sys
import termios
import tty
import serial


class MyTeleop:
    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=9600):
        self.lspeed = 0
        self.rspeed = 0

        if serial_port is None:
            raise ValueError("Для serial-режима необходимо указать serial_port")
        self.ser = serial.Serial(serial_port, baudrate, timeout=1)

        #self.orig_settings = termios.tcgetattr(sys.stdin)
        #tty.setcbreak(sys.stdin)

    def get_speed(self):
        return f"левая скорость: {self.lspeed} \nправая скорость: {self.rspeed}"


    @staticmethod
    def voltage_to_bytes_msg(left_voltage: int, right_voltage: int) -> bytes:
        """
        Converts the voltages of the left and right motors into a byte message.

        :param left_voltage: Voltage of the left motor (from 0 to 255).
        :param right_voltage: Voltage of the right motor (from 0 to 255).
        ::return: 3-byte message.

        orientation = 0 left +, right + : moving forward
        orientation = 1 left -, right - : moving backward
        orientation = 2 left +, right - : turning right
        orientation = 3 left -, right + : turning left
        """
        orientation = 0

        if left_voltage < 0 and right_voltage < 0:
            orientation = 1

        elif left_voltage >= 0 > right_voltage:
            orientation = 2

        elif left_voltage < 0 <= right_voltage:
            orientation = 3

        left_motor = abs(left_voltage) if abs(left_voltage) <= 255 else 255
        right_motor = abs(right_voltage) if abs(right_voltage) <= 255 else 255
        message = struct.pack("BBB", left_motor, right_motor, orientation)

        return message

    def send_velocity(self, left_voltage, right_voltage):
        self.lspeed = left_voltage
        self.rspeed = right_voltage
        msg = MyTeleop.voltage_to_bytes_msg(left_voltage, right_voltage)
        self.ser.write(msg)


    def run(self):
        try:
            pass
            #while True:
             #pass
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)
        self.ser.close()
