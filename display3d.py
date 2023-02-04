from glumpy import app, gloo, gl, glm, __version__
from glumpy.transforms import Position, Trackball
import numpy as np

class Display3D:
    def __init__(self, width, height):
        self.n = 0
        
        # Build the program and corresponding buffers (with 4 vertices)
        self.vertex = """
            attribute vec3 position;
            attribute vec3 color;
            varying vec3   v_color;
            varying vec4  v_eye_position;
            
            void main(){
                //v_eye_position = <transform.trackball_view> *
                //               <transform.trackball_model> *
                //                vec4(position,1.0);

                v_color = color;
                gl_Position = <transform(vec4(position, 1.0))>;
                gl_PointSize = 8;

                // stackoverflow.com/questions/8608844/...
                //  ... resizing-point-sprites-based-on-distance-from-the-camera
                //vec4 p = <transform.trackball_projection> *
                //         vec4(radius, radius, v_eye_position.z, v_eye_position.w);
                //v_size = 512.0 * p.x / p.w;
                //gl_PointSize = v_size + 5.0;
            }
        """

        self.fragment = """
            varying vec3      v_color;
            //varying vec4 v_eye_position;

            void main() {
                gl_FragColor = vec4(v_color, 0.3);
            } 
        """

        # Window and opengl context
        self.window = app.Window(width=int(width), height=int(height), color=(0.1,0.1,0.1,1))
        #self.console = app.Console(rows=32, cols=80, color=(1,1,1,1))
        #self.window.attach(self.console)

        # vertext buffer
        self.data = np.zeros(50000, dtype = [ ("position", np.float32, 3),
                                          ("color",    np.float32, 3)] )
        self.data = self.data.view(gloo.VertexBuffer)
        
        # shader program
        self.program = gloo.Program(self.vertex, self.fragment)
        self.program.bind(self.data)

        # transform
        transform = Trackball(Position("position"), aspect=1)
        #transform['theta'] = 0
        #transform['phi'] = 0
        self.program["transform"] = transform
        self.window.attach(transform)
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        #gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        #gl.glPointSize(4)

        # window events
        @self.window.event
        def on_draw(dt):
            self.window.clear()
            self.program.draw(gl.GL_POINTS)
            #self.console.draw()

    def add_point(self, pt, color):
        pt[2] = -pt[2] # opengl neg-z axis into scene
        self.data['position'][self.n] = pt
        self.data['color'][self.n] = color
        self.n += 1
        self.program.bind(self.data)


