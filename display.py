from glumpy import app, gloo, gl, __version__

class Display:
    def __init__(self, width, height):
        # Build the program and corresponding buffers (with 4 vertices)
        self.vertex = """
            attribute vec2 position;
            attribute vec2 a_texcoord;
            varying vec2   v_texcoord;
            
            void main(){
                v_texcoord  = a_texcoord;
                gl_Position = vec4(position, 0.0, 1.0);
            }
        """

        self.fragment = """
            uniform sampler2D u_texture;
            varying vec2      v_texcoord;

            void main() {
                gl_FragColor = texture2D(u_texture, v_texcoord);
            } 
        """

        self.quad = gloo.Program(self.vertex, self.fragment, count=4)

        self.console = app.Console(rows=32, cols=80, color=(1,1,1,1))

        # Create a window with a valid GL context
        self.window = app.Window(width=int(width), height=int(height), color=(1,1,1,1))
        self.window.attach(self.console)

        # Upload data into GPU
        self.quad['position'] = (-1,+1), (-1,-1), (+1,+1), (+1,-1)
        self.quad['a_texcoord'] = (0,0), (0,1), (1,0), (1,1)

        @self.window.event
        def on_draw(dt):
            self.window.clear()
            self.quad.draw(gl.GL_TRIANGLE_STRIP)
            self.console.draw()

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == 32:
                return

    def set_image(self, image):
        self.quad['u_texture'] = image

