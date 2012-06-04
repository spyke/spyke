import sys
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt

class TestWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.list = TestListView(self)
        self.setCentralWidget(self.list)
        self.resize(500, 500)

class TestListView(QtGui.QListView):
    def __init__(self, parent):
        QtGui.QListView.__init__(self, parent)
        self.setModel(TestListModel(parent))
        self.setUniformItemSizes(True) # speeds up listview
        self.setFlow(QtGui.QListView.LeftToRight) # default is TopToBottom
        self.setWrapping(True) # default is False

class TestListModel(QtCore.QAbstractListModel):
    def __init__(self, parent):
        QtCore.QAbstractListModel.__init__(self, parent)

    def rowCount(self, parent=None):
        return 1000

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            return index.row()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    win = TestWindow()
    win.show()
    sys.exit(app.exec_())


"""
QListView: keyboard navigation doesn't wrap when Wrapping is enabled

Summary: with wrapping enabled in a QListView, if the cursor is at an edge of the list, keyboard navigation toward that edge does nothing.

Here's a wordy but more thorough description:

In TopToBottom flow mode with wrapping enabled, when the cursor is at the bottom of a column, hitting down doesn't navigate to the top of the next column. Similarly, when at the top of a column, hitting up doesn't navigate to the bottom of the previous column.

In LeftToRight flow mode with wrapping enabled, when the cursor is at the right edge of a row, hitting right doesn't navigate to the left edge of the next row. Similarly, when at the left edge of a row, hitting left doesn't navigate to the right edge of the previous row.

(Note that when I use the terms "row" and "column", these are sort of pseudo terms. With wrapping turned on, they look like rows and columns, but of course the list model itself is nothing but rows.)

This was mentioned by two other posters, with no responses:
http://www.qtcentre.org/threads/21681-QListView-keyboard-nav
http://www.qtcentre.org/threads/35010-Navigation-by-QListView-with-enabled-isWrapping-property

I'm using the stock Qt 4.7.0 with PyQt 4.7.4 from Ubuntu 10.10 64-bit repos. Here's a minimal python program that demonstrates this behaviour:




I'd argue that this behvaviour is unexpected. Compact view and icon view in any file manager I've ever used wraps the cursor if it reaches an edge and if there's logically somewhere for it to go next. This makes the most of keyboard navigation.

Might this be considered a bug, or an unimplemented feature? Anyone know of any workarounds? The use case that I personally care about is LeftToRight flow mode with Wrapping enabled. I tried overriding QListView.KeyPressEvent() for left and right keys, like this:

def keyPressEvent(self, event):
    if event.key() == Qt.Key_Left:
        self.setCurrentIndex(self.moveCursor(self.MovePrevious, event.modifiers()))
    elif event.key() == Qt.Key_Right:
        self.setCurrentIndex(self.moveCursor(self.MoveNext, event.modifiers()))
    else:
        QtGui.QListView.keyPressEvent(self, event) # handle it as usual

but that just moves the cursor up and down, not left and right. Using MoveLeft and MoveRight does what it says, but again doesn't wrap around the left or right edge, which is my original problem. Looking at the code in http://qt.gitorious.org/qt/qt/blobs/master/src/gui/itemviews/qlistview.cpp, "case MovePrevious" and "case MoveNext" are both empty in QListView::moveCursor. They both precede "case MoveUp" and "case MoveDown", which I presume in C++ means that's what they default to.

Cheers,

Martin
"""
