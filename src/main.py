from tkinter import *
from PIL import Image, ImageDraw, ImageFilter
from numpy import asarray
import numpy as np
import cv2
from cnn_model.cnn_model import CnnModel


class Paint(object):
    # parameters of the drawing
    PEN_SIZE = 56
    BG_COLOR = 'black'
    PEN_COLOR = 'white'
    IMG_WIDTH = 560  # 28x20
    IMG_HEIGHT = 560

    def __init__(self):
        # Creates the GUI window
        self.window = Tk()
        self.window.title('Handwritten number prediction from 0 to 9')

        self.window.columnconfigure(0, weight=1, minsize=80)
        self.window.columnconfigure(1, weight=1, minsize=80)
        self.window.rowconfigure(0, weight=1, minsize=80)
        self.window.rowconfigure(1, weight=1, minsize=80)

        # Needs picture 28x28 or scaled up for the ML model
        self.image = Image.new('L', (self.IMG_WIDTH, self.IMG_HEIGHT))
        self.draw = ImageDraw.Draw(self.image)

        # self.image is used to show visually what is being drawn and canvas are used to save that image
        self.cv = Canvas(self.window, bg=self.BG_COLOR, width=self.IMG_WIDTH, height=self.IMG_HEIGHT)
        self.cv.grid(row=0, column=0)
        # make sure that the size of the display is correct, it was 430 in the notebook
        self.display = Text(self.window, height=1, width=1, font=('Helvetica', 370))
        self.display.grid(row=0, column=1)

        # Set up the clear button
        self.clear = Button(text='Clear', width=20, height=5, command=self.clear_img)
        self.clear.grid(row=1, column=0)

        # Set up the predict button
        self.predict = Button(text='Predict', width=20, height=5, command=self.predict_img)
        self.predict.grid(row=1, column=1)

        self.old_x = None
        self.old_y = None

        # Binding the motion of the mouse and clicking to the canvas to allow drawing
        self.cv.bind('<B1-Motion>', self.paint)
        self.cv.bind('<ButtonRelease-1>', self.reset)

        self.window.mainloop()

    def paint(self, event):
        """
        Function that draws a line in response to the mouse movement and holding down the mouse button
        :param event: Cick of the mouse button, here used specifically for the coordinates of that click on the canvas
        """
        if self.old_x and self.old_y:
            self.cv.create_line(self.old_x, self.old_y, event.x, event.y,
                                width=self.PEN_SIZE, fill=self.PEN_COLOR,
                                capstyle=ROUND, smooth=TRUE, splinesteps=36)
            # rename self.draw later
            self.draw.line([(self.old_x, self.old_y), (event.x, event.y)],
                           fill=self.PEN_COLOR, width=self.PEN_SIZE)

        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        """
        Utility function that resets the coordinates
        :param event: Click of the button
        """
        self.old_x, self.old_y = None, None

    def preprocess(self, image):
        """
        Utility function that filters the image and resizes and reshapes it for the use in the cnn model
        :param image: Raw image painted by the user
        :return: Filtered image in the correct size and shape for the cnn model
        """
        image = image.filter(ImageFilter.MaxFilter(9))
        data = asarray(image).copy().astype('float32')
        return cv2.resize(data, dsize=(28, 28), interpolation=cv2.INTER_CUBIC).reshape(1, 28, 28, 1)

    def clear_img(self):
        """
        Utility function that clears the user's drawing by drawing a rectangle of the same size as the canvas
        """
        self.cv.delete('all')
        self.draw.rectangle((0, 0, self.IMG_WIDTH, self.IMG_HEIGHT), fill=self.BG_COLOR)
        self.display.delete('1.0', END)

    def predict_img(self):
        """
        Utility function that clears the display, preprocesses the image, gets the prediction and displays it
        """
        self.display.delete('1.0', END)
        image = self.preprocess(self.image)
        prediction = np.argmax(model.predict(image).flatten())
        self.display.insert('1.0', prediction)


if __name__ == '__main__':
    model = CnnModel().build_model()

    try:
        model.load_weights('./data/model_weights')
    except:
        model.compile_train_model()

    Paint()