"""
2D rendering framework
"""
import math
import os
import sys

import numpy as np
from gym import error
from pyglet.gl import (
    GL_BLEND,
    GL_LINE_LOOP,
    GL_LINE_SMOOTH,
    GL_LINE_SMOOTH_HINT,
    GL_LINE_STIPPLE,
    GL_LINE_STRIP,
    GL_LINES,
    GL_NICEST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_POLYGON,
    GL_QUADS,
    GL_SRC_ALPHA,
    GL_TRIANGLES, glFlush, glMatrixMode, GL_MODELVIEW,
)

try:
    import pyglet
except ImportError:
    raise ImportError(
        "HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it."
    )

try:
    from pyglet.gl import (
        glBegin,
        glBlendFunc,
        glClearColor,
        glColor4f,
        glDisable,
        glEnable,
        glEnd,
        glHint,
        glLineStipple,
        glLineWidth,
        glPopMatrix,
        glPushMatrix,
        glRotatef,
        glScalef,
        glTranslatef,
        gluOrtho2D,
        glVertex2f,
        glVertex3f,
    )
except ImportError:
    raise ImportError(
        """Error occurred while running `from pyglet.gl import ...`
            HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'"""
    )


if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"


RAD2DEG = 57.29577951308232


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            f"Invalid display specification: {spec}. (Must be a string like :0 or None.)"
        )


class Viewer:
    def __init__(self, width, height, display=None):
        try:
            # 尝试硬件加速
            config = pyglet.gl.Config(double_buffer=True)
            self.window = pyglet.window.Window(config=config, visible=False)
        except pyglet.window.NoSuchConfigException:
            # 回退到软件渲染
            config = pyglet.gl.Config(double_buffer=True, driver='software')
            self.window = pyglet.window.Window(config=config, visible=False)
        display = get_display(display)

        self.width = width
        self.height = height

        # self.window = pyglet.window.Window(width=width, height=height, display=display)
        # 添加OpenGL配置
        config = pyglet.gl.Config(
            double_buffer=True,
            depth_size=24,
            major_version=2,  # 使用兼容的OpenGL 2.1
            minor_version=1
        )
        self.window = pyglet.window.Window(
            width=width,
            height=height,
            display=display,
            config=config,  # 添加双缓冲配置
            visible=True  # 强制窗口可见
        )
        self.window.set_visible(True)  # 强制窗口可见
        self.window.switch_to()  # 立即激活上下文
        self.window.dispatch_events()  # 立即处理事件创建上下文
        self.window.on_close = self.window_closed_by_user
        self.geoms = []
        self.onetime_geoms = []
        self.text_lines = []
        self.transform = Transform()
        self.max_size = 1

        glEnable(GL_BLEND)
        # glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_max_size(self, current_size):
        max_size = self.max_size = max(current_size, self.max_size)
        left = -max_size
        right = max_size
        bottom = -max_size
        top = max_size
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()

        pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
        pyglet.gl.glLoadIdentity()
        gluOrtho2D(0, self.window.width, 0, self.window.height)
        for geom in self.text_lines:
            geom.render()

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr

    # Convenience
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        self.window.flip()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1, :, 0:3]


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


class Geom:
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)


class Attr:
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    # def enable(self):
    #     glPushMatrix()
    #     glTranslatef(
    #         self.translation[0], self.translation[1], 0
    #     )  # translate to GL loc ppint
    #     glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
    #     glScalef(self.scale[0], self.scale[1], 1)

    def enable(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(*self.translation, 0)
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(*self.scale, 1)
    def disable(self):
        glPopMatrix()
        # 添加显式刷新指令
        glFlush()
    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        glColor4f(*self.vec4)


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style

    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)

    def disable(self):
        glDisable(GL_LINE_STIPPLE)


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        glLineWidth(self.stroke)


class Point(Geom):
    def __init__(self):
        Geom.__init__(self)

    def render1(self):
        (GL_POINTS)  # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()


class TextLine:
    def __init__(self, window, idx):
        self.idx = idx
        self.window = window
        pyglet.font.add_file(os.path.join(os.path.dirname(__file__), "secrcode.ttf"))
        self.label = None
        self.set_text("")

    def render(self):
        if self.label is not None:
            self.label.draw()

    def set_text(self, text):
        if pyglet.font.have_font("Courier"):
            font = "Courier"
        elif pyglet.font.have_font("Secret Code"):
            font = "Secret Code"
        else:
            return

        self.label = pyglet.text.Label(
            text,
            font_name=font,
            color=(0, 0, 0, 255),
            font_size=20,
            x=0,
            y=self.idx * 40 + 20,
            anchor_x="left",
            anchor_y="bottom",
        )

        self.label.draw()


class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def render1(self):
        glPushMatrix()  # 添加矩阵保护
        if len(self.v) == 4:
            glBegin(GL_QUADS)
        elif len(self.v) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()
        glPopMatrix()  # 恢复矩阵状态

        color = (
            self._color.vec4[0] * 0.5,
            self._color.vec4[1] * 0.5,
            self._color.vec4[2] * 0.5,
            self._color.vec4[3] * 0.5,
        )
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()


def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_polygon(v, filled=True):
    if filled:
        return FilledPolygon(v)
    else:
        return PolyLine(v, True)


def make_polyline(v):
    return PolyLine(v, False)


def make_capsule(length, width):
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render1(self):
        for g in self.gs:
            g.render()


class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()

    def set_linewidth(self, x):
        self.linewidth.stroke = x


class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()


class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(
            -self.width / 2, -self.height / 2, width=self.width, height=self.height
        )


class SimpleImageViewer:
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display
            )
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (
            self.height,
            self.width,
            3,
        ), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(
            self.width, self.height, "RGB", arr.tobytes(), pitch=self.width * -3
        )
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


# rendering.py 新增函数
def make_rectangle(width, height, filled=True):
    from gym.envs.classic_control import rendering
    l, r, t, b = -width/2, width/2, -height/2, height/2
    rect = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) if filled else \
           rendering.PolyLine([(l,b), (l,t), (r,t), (r,b)], True)
    rect.add_attr(rendering.Transform())
    return rect