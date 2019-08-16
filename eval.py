import torch
from matplotlib import pyplot as plt
import argparse
from kivy.app import App
from kivy.uix.button import Button

class TestApp(App):
    def build(self):
        return Button(text='Hello World')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model evaluation utility')
    parser.add_argument('-d', required=True,
                        help='folder containing data to be used for evaluation')
    parser.add_argument('--show', dest='s', action='store_true', default=False,
                        help='show examples')
    parser.add_argument('--interactive', '-i', dest='i', action='store_true', default=False,
                        help='show examples')
    args = parser.parse_args()

    if args.s:
        print("vizualizing results now")

    if args.i:
        TestApp().run()