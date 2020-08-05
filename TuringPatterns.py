# Modules
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt 
from matplotlib import animation
import numpy as np
import seaborn as sns
import threading
sns.set_style("whitegrid")

# plot_time_evolution and plot_evolution_outcome definitions
class BaseStateSystem:
    """
    Base object for "State System".
    We are going to repeatedly visualise systems which are Markovian:
    the have a "state", the state evolves in discrete steps, and the next
    state only depends on the previous state.
    To make things simple, I'm going to use this class as an interface.
    """
    def __init__(self):
        raise NotImplementedError()

    def initialise(self):
        raise NotImplementedError()

    def initialise_figure(self):
        fig, ax = plt.subplots()
        return fig, ax

    def update(self):
        raise NotImplementedError()

    def draw(self, ax):
        raise NotImplementedError()

    def plot_time_evolution(self, filename, n_steps=30):
        """
        Creates a gif from the time evolution of a basic state syste.
        """
        self.initialise()
        fig, ax = self.initialise_figure()

        def step(t):
            self.update()
            self.draw(ax)

        anim = animation.FuncAnimation(fig, step, frames=np.arange(n_steps), interval=20)
        anim.save(filename=filename, dpi=60, fps=10)
        plt.close()
        
    def plot_evolution_outcome(self, filename, n_steps):
        """
        Evolves and save the outcome of evolving the system for n_steps
        """
        self.initialise()
        fig, ax = self.initialise_figure()
        
        for _ in range(n_steps):
            self.update()

        self.draw(ax)
        fig.savefig(filename)
        plt.close()

def laplacian1D(a, dx):
    return (
        - 2 * a
        + np.roll(a,1,axis=0) 
        + np.roll(a,-1,axis=0)
    ) / (dx ** 2)

def laplacian2D(a, dx):
    return (
        - 4 * a
        + np.roll(a,1,axis=0) 
        + np.roll(a,-1,axis=0)
        + np.roll(a,+1,axis=1)
        + np.roll(a,-1,axis=1)
    ) / (dx ** 2)


# Diffusion model
def startDiffusion():
    progressText.config(text = "Your vizualisation will be downloaded as soon as modelling is finished, please note longer generation models and gifs may take longer to model. Thank You.")
    global thrd1
    thrd1 = threading.Thread(target = getDiffusionGif)
    thrd1.start()

def getDiffusionGif():
        
    class OneDimensionalDiffusionEquation(BaseStateSystem):
        def __init__(self, D):
            self.D = D
            self.width = 1000
            self.dx = 10 / self.width
            self.dt = 0.9 * (self.dx ** 2) / (2 * D)
            self.steps = int(0.1 / self.dt)
            
        def initialise(self):
            self.t = 0
            self.X = np.linspace(-5,5,self.width)
            self.a = np.exp(-self.X**2)
            
        def update(self):
            for _ in range(self.steps):
                self.t += self.dt
                self._update()

        def _update(self):      
            La = laplacian1D(self.a, self.dx)
            delta_a = self.dt * (self.D * La)       
            self.a += delta_a
            
        def draw(self, ax):
            ax.clear()
            ax.plot(self.X,self.a, color="r")
            ax.set_ylim(0,1)
            ax.set_xlim(-5,5)
            ax.set_title("t = {:.2f}".format(self.t)) 

    one_d_diffusion = OneDimensionalDiffusionEquation(D=1)
    one_d_diffusion.plot_time_evolution("diffusion.gif")
    progressText.config(text = "Modelling and download complete, ready for next visualisation")

# Reaction model
def startReaction(a, b):
    progressText.config(text = "Your vizualisation will be downloaded as soon as modelling is finished, please note longer generation models and gifs may take longer to model. Thank You.")
    global thrd2
    thrd2 = threading.Thread(target = getReactionGif, args = [a, b])
    thrd2.start()

def getReactionGif(a, b): # 0 < Initial concentrations < 1
    class ReactionEquation(BaseStateSystem):
        def __init__(self, Ra, Rb):
            self.Ra = Ra
            self.Rb = Rb
            self.dt = 0.01
            self.steps = int(0.1 / self.dt)
            
        def initialise(self):
            self.t = 0
            self.a = a
            self.b = b
            self.Ya = []
            self.Yb = []
            self.X = []
            
        def update(self):
            for _ in range(self.steps):
                self.t += self.dt
                self._update()

        def _update(self):      
            delta_a = self.dt * self.Ra(self.a,self.b)      
            delta_b = self.dt * self.Rb(self.a,self.b)      

            self.a += delta_a
            self.b += delta_b
            
        def draw(self, ax):
            ax.clear()
            
            self.X.append(self.t)
            self.Ya.append(self.a)
            self.Yb.append(self.b)

            ax.plot(self.X,self.Ya, color="r", label="A")
            ax.plot(self.X,self.Yb, color="b", label="B")
            ax.legend()
            
            ax.set_ylim(0,1)
            ax.set_xlim(0,5)
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentrations")
            
    alpha, beta =  0.2, 5

    def Ra(a,b): return a - a ** 3 - b + alpha
    def Rb(a,b): return (a - b) * beta
        
    one_d_reaction = ReactionEquation(Ra, Rb)
    one_d_reaction.plot_time_evolution("reaction.gif", n_steps=50)

    progressText.config(text = "Modelling and download complete, ready for next visualisation")

# Random generating function
def random_initialiser(shape):
    return(
        np.random.normal(loc=0, scale=0.05, size=shape),
        np.random.normal(loc=0, scale=0.05, size=shape)
    )

# 1D stills
def start1Dpng(generations):
    progressText.config(text = "Your vizualisation will be downloaded as soon as modelling is finished, please note longer generation models and gifs may take longer to model. Thank You.")
    global thrd3
    thrd3 = threading.Thread(target = get1Dpng, args = [generations])
    thrd3.start()

def get1Dpng(generations):

    class OneDimensionalRDEquations(BaseStateSystem):
        def __init__(self, Da, Db, Ra, Rb,
                    initialiser=random_initialiser,
                    width=1000, dx=1, 
                    dt=0.1, steps=1):
            
            self.Da = Da
            self.Db = Db
            self.Ra = Ra
            self.Rb = Rb
            
            self.initialiser = initialiser
            self.width = width
            self.dx = dx
            self.dt = dt
            self.steps = steps
            
        def initialise(self):
            self.t = 0
            self.a, self.b = self.initialiser(self.width)
            
        def update(self):
            for _ in range(self.steps):
                self.t += self.dt
                self._update()

        def _update(self):
            
            # unpack so we don't have to keep writing "self"
            a,b,Da,Db,Ra,Rb,dt,dx = (
                self.a, self.b,
                self.Da, self.Db,
                self.Ra, self.Rb,
                self.dt, self.dx
            )
            
            La = laplacian1D(a, dx)
            Lb = laplacian1D(b, dx)
            
            delta_a = dt * (Da * La + Ra(a,b))
            delta_b = dt * (Db * Lb + Rb(a,b))
            
            self.a += delta_a
            self.b += delta_b
            
        def draw(self, ax):
            ax.clear()
            ax.plot(self.a, color="r", label="A")
            ax.plot(self.b, color="b", label="B")
            ax.legend()
            ax.set_ylim(-1,1)
            ax.set_title("t = {:.2f}".format(self.t))
            
    Da, Db, alpha, beta = 1, 100, -0.005, 10

    def Ra(a,b): return a - a ** 3 - b + alpha
    def Rb(a,b): return (a - b) * beta

    width = 100
    dx = 1
    dt = 0.001

    OneDimensionalRDEquations(
        Da, Db, Ra, Rb, 
        width=width, dx=dx, dt=dt, 
        steps=100
    ).plot_evolution_outcome("1dRD.png", n_steps=generations)
    progressText.config(text = "Modelling and download complete, ready for next visualisation")

# 1D gif
def start1Dgif(generations):
    progressText.config(text = "Your vizualisation will be downloaded as soon as modelling is finished, please note longer generation models and gifs may take longer to model. Thank You.")
    global thrd4
    thrd4 = threading.Thread(target = get1Dgif, args = [generations])
    thrd4.start()

def get1Dgif(generations):
    class OneDimensionalRDEquations(BaseStateSystem):
        def __init__(self, Da, Db, Ra, Rb,
                    initialiser=random_initialiser,
                    width=1000, dx=1, 
                    dt=0.1, steps=1):
            
            self.Da = Da
            self.Db = Db
            self.Ra = Ra
            self.Rb = Rb
            
            self.initialiser = initialiser
            self.width = width
            self.dx = dx
            self.dt = dt
            self.steps = steps
            
        def initialise(self):
            self.t = 0
            self.a, self.b = self.initialiser(self.width)
            
        def update(self):
            for _ in range(self.steps):
                self.t += self.dt
                self._update()

        def _update(self):
            
            # unpack so we don't have to keep writing "self"
            a,b,Da,Db,Ra,Rb,dt,dx = (
                self.a, self.b,
                self.Da, self.Db,
                self.Ra, self.Rb,
                self.dt, self.dx
            )
            
            La = laplacian1D(a, dx)
            Lb = laplacian1D(b, dx)
            
            delta_a = dt * (Da * La + Ra(a,b))
            delta_b = dt * (Db * Lb + Rb(a,b))
            
            self.a += delta_a
            self.b += delta_b
            
        def draw(self, ax):
            ax.clear()
            ax.plot(self.a, color="r", label="A")
            ax.plot(self.b, color="b", label="B")
            ax.legend()
            ax.set_ylim(-1,1)
            ax.set_title("t = {:.2f}".format(self.t))
            
    Da, Db, alpha, beta = 1, 100, -0.005, 10

    def Ra(a,b): return a - a ** 3 - b + alpha
    def Rb(a,b): return (a - b) * beta

    width = 100
    dx = 1
    dt = 0.001

    OneDimensionalRDEquations(
        Da, Db, Ra, Rb, 
        width=width, dx=dx, dt=dt, 
        steps=100
    ).plot_time_evolution("1dRD.gif", n_steps=generations)
    progressText.config(text = "Modelling and download complete, ready for next visualisation")

# 2D still
def start2Dpng(generations):
    progressText.config(text = "Your vizualisation will be downloaded as soon as modelling is finished, please note longer generation models and gifs may take longer to model. Thank You.")
    global thrd5
    thrd5 = threading.Thread(target = get2Dpng, args = [generations])
    thrd5.start()

def get2Dpng(generations):
    class TwoDimensionalRDEquations(BaseStateSystem):
        def __init__(self, Da, Db, Ra, Rb,
                    initialiser=random_initialiser,
                    width=1000, height=1000,
                    dx=1, dt=0.1, steps=1):
            
            self.Da = Da
            self.Db = Db
            self.Ra = Ra
            self.Rb = Rb

            self.initialiser = initialiser
            self.width = width
            self.height = height
            self.shape = (width, height)
            self.dx = dx
            self.dt = dt
            self.steps = steps
            
        def initialise(self):
            self.t = 0
            self.a, self.b = self.initialiser(self.shape)
            
        def update(self):
            for _ in range(self.steps):
                self.t += self.dt
                self._update()

        def _update(self):
            
            # unpack so we don't have to keep writing "self"
            a,b,Da,Db,Ra,Rb,dt,dx = (
                self.a, self.b,
                self.Da, self.Db,
                self.Ra, self.Rb,
                self.dt, self.dx
            )
            
            La = laplacian2D(a, dx)
            Lb = laplacian2D(b, dx)
            
            delta_a = dt * (Da * La + Ra(a,b))
            delta_b = dt * (Db * Lb + Rb(a,b))
            
            self.a += delta_a
            self.b += delta_b
            
        def draw(self, ax):
            ax[0].clear()
            ax[1].clear()

            ax[0].imshow(self.a, cmap='jet')
            ax[1].imshow(self.b, cmap='brg')
            
            ax[0].grid(b=False)
            ax[1].grid(b=False)
            
            ax[0].set_title("A, t = {:.2f}".format(self.t))
            ax[1].set_title("B, t = {:.2f}".format(self.t))
            
        def initialise_figure(self):
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
            return fig, ax
        
    Da, Db, alpha, beta = 1, 100, -0.005, 10

    def Ra(a,b): return a - a ** 3 - b + alpha
    def Rb(a,b): return (a - b) * beta

    width = 100
    dx = 1
    dt = 0.001

    TwoDimensionalRDEquations(
        Da, Db, Ra, Rb, 
        width=width, height=width, 
        dx=dx, dt=dt, steps=100
    ).plot_evolution_outcome("2dRD.png", n_steps=generations)
    progressText.config(text = "Modelling and download complete, ready for next visualisation")

# 2D gif
def start2Dgif(generations):
    progressText.config(text = "Your vizualisation will be downloaded as soon as modelling is finished, please note longer generation models and gifs may take longer to model. Thank You.")
    global thrd6
    thrd6 = threading.Thread(target = get2Dgif, args = [generations])
    thrd6.start()

def get2Dgif(generations):
    class TwoDimensionalRDEquations(BaseStateSystem):
        def __init__(self, Da, Db, Ra, Rb,
                    initialiser=random_initialiser,
                    width=1000, height=1000,
                    dx=1, dt=0.1, steps=1):
            
            self.Da = Da
            self.Db = Db
            self.Ra = Ra
            self.Rb = Rb

            self.initialiser = initialiser
            self.width = width
            self.height = height
            self.shape = (width, height)
            self.dx = dx
            self.dt = dt
            self.steps = steps
            
        def initialise(self):
            self.t = 0
            self.a, self.b = self.initialiser(self.shape)
            
        def update(self):
            for _ in range(self.steps):
                self.t += self.dt
                self._update()

        def _update(self):
            
            # unpack so we don't have to keep writing "self"
            a,b,Da,Db,Ra,Rb,dt,dx = (
                self.a, self.b,
                self.Da, self.Db,
                self.Ra, self.Rb,
                self.dt, self.dx
            )
            
            La = laplacian2D(a, dx)
            Lb = laplacian2D(b, dx)
            
            delta_a = dt * (Da * La + Ra(a,b))
            delta_b = dt * (Db * Lb + Rb(a,b))
            
            self.a += delta_a
            self.b += delta_b
            
        def draw(self, ax):
            ax[0].clear()
            ax[1].clear()

            ax[0].imshow(self.a, cmap='jet')
            ax[1].imshow(self.b, cmap='brg')
            
            ax[0].grid(b=False)
            ax[1].grid(b=False)
            
            ax[0].set_title("A, t = {:.2f}".format(self.t))
            ax[1].set_title("B, t = {:.2f}".format(self.t))
            
        def initialise_figure(self):
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
            return fig, ax
        
    Da, Db, alpha, beta = 1, 100, -0.005, 10

    def Ra(a,b): return a - a ** 3 - b + alpha
    def Rb(a,b): return (a - b) * beta

    width = 100
    dx = 1
    dt = 0.001

    TwoDimensionalRDEquations(
        Da, Db, Ra, Rb, 
        width=width, height=width, 
        dx=dx, dt=dt, steps=100
    ).plot_time_evolution("2dRD.gif", n_steps=generations)
    progressText.config(text = "Modelling and download complete, ready for next visualisation")


# GUI, GUI, AND MORE GUI
# Fonts
HEADING = ("Verdana 12 bold") # Should be centered
SUBHEADING = ("Verdana 10 italic") # Should be left

# Gui setup, frame dictionary setup
class Gui(tk.Tk): 
    def __init__(self, *args, **kwargs):  
        tk.Tk.__init__(self, *args, **kwargs) 
          
        container = tk.Frame(self)   
        container.pack(side = "top", fill = "both", expand = True)  
        container.grid_rowconfigure(0, weight = 1) 
        container.grid_columnconfigure(0, weight = 1) 
   
        self.frames = {}   
   
        for F in [Home, Start]: 
            frame = F(container, self) 
            self.frames[F] = frame  
            frame.grid(row = 0, column = 0, sticky ="nsew") 
   
        self.show_frame(Home) 

    def show_frame(self, cont): 
        frame = self.frames[cont] 
        frame.tkraise() 


class Home(tk.Frame): 
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent)

        homeFrame = ttk.LabelFrame(self, text = "Home")
        homeFrame.pack(padx = 10, pady = 5, expand = True)

        homeTitle = ttk.Label(homeFrame, text = "Turing Pattern Visualiser v0.1", font = HEADING)
        homeTitle.grid(row = 0, column = 0, columnspan = 2, padx = 10, pady = 5)

        homeText = ttk.Label(homeFrame, text = "This tool is used for visualing Turing Patterns using mechanisms suggested in his 1952 'The Chemical Basis of Morphogenesis'\n\nMethodology and Matplotlib implementation: Iain Barr - https://github.com/ijmbarr/turing-patterns or http://www.degeneratestate.org/posts/2017/May/05/turing-patterns/ \nGUI Page Design modelled on: https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter \nDesign: Angus Toms - (angus.toms@icloud.com) \nAugust 5th, 2020", wraplength = 600, justify = tk.CENTER)
        homeText.grid(row = 1, column = 0, columnspan = 2, padx = 10, pady = 5)

        homeExitButton = ttk.Button(homeFrame, text = "Exit", command = tryExit)
        homeExitButton.grid(row = 2, column = 0, padx = 10, pady = 5)

        homeStartButton = ttk.Button(homeFrame, text = "Start", command = lambda: controller.show_frame(Start))
        homeStartButton.grid(row = 2, column = 1, padx = 10, pady = 5)

class Start(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        startFrame = ttk.LabelFrame(self, text = "Create visualisations")
        startFrame.pack(padx = 10, pady = 10, expand = True)

        startTitle = ttk.Label(startFrame, text = "Select the visualisation you would like to create.", font = HEADING)
        startTitle.grid(row = 0, column = 0, columnspan = 2, padx = 10, pady = 5)

        diffusionFrame = ttk.LabelFrame(startFrame, text = "Diffusion Waves")
        diffusionFrame.grid(row = 1, column = 0, padx = 10, pady = 5, sticky = "nsew")

        diffusionGo = ttk.Button(diffusionFrame, text = "Go", command = getDiffusionGif)
        diffusionGo.pack(padx = 10, pady = 5, expand = True)

        reactionFrame = ttk.LabelFrame(startFrame, text = "Reaction Waves")
        reactionFrame.grid(row = 1, column = 1, padx = 10, pady = 5, sticky = "nsew")

        concentrationLabelA = ttk.Label(reactionFrame, text = "Concentration of A:")
        concentrationLabelA.grid(row = 0, column = 0, padx = 5, pady = 2.5, sticky = "e")

        concentrationLabelB = ttk.Label(reactionFrame, text = "Concentration of B:")
        concentrationLabelB.grid(row = 1, column = 0, padx = 5, pady = 2.5, sticky = "e")

        global aEntry
        aEntry = ttk.Entry(reactionFrame)
        aEntry.grid(row = 0, column = 1, padx = 5, pady = 2.5)

        global bEntry
        bEntry = ttk.Entry(reactionFrame)
        bEntry.grid(row = 1, column = 1, padx = 5, pady = 2.5)

        reactionGo = ttk.Button(reactionFrame, text = "Go", command = lambda: startReaction(int(aEntry.get()), int(bEntry.get())))
        reactionGo.grid(row = 2, column = 0, columnspan = 2, padx = 5, pady = 2.5)

        onePngFrame = ttk.LabelFrame(startFrame, text = "One Dimensional PNG")
        onePngFrame.grid(row = 2, column = 0, padx = 10, pady = 5, sticky = "nsew")

        onePngText = ttk.Label(onePngFrame, text = "Generations:")
        onePngText.grid(row = 0, column = 0, padx = 5, pady = 2.5, sticky = "e")

        global onePngEntry
        onePngEntry = ttk.Entry(onePngFrame)
        onePngEntry.grid(row = 0, column = 1, padx = 5, pady = 2.5)

        onePngGo = ttk.Button(onePngFrame, text = "Go", command = lambda: start1Dpng(int(onePngEntry.get())))
        onePngGo.grid(row = 1, column = 0, columnspan = 2, padx = 5, pady = 2.5)

        oneGifFrame = ttk.LabelFrame(startFrame, text = "One Dimensional GIF")
        oneGifFrame.grid(row = 2, column = 1, padx = 10, pady = 5)

        oneGifText = ttk.Label(oneGifFrame, text = "Generations:")
        oneGifText.grid(row = 0, column = 0, padx = 5, pady = 2.5, sticky = "e")

        global oneGifEntry
        oneGifEntry = ttk.Entry(oneGifFrame)
        oneGifEntry.grid(row = 0, column = 1, padx = 5, pady = 2.5)

        oneGifGo = ttk.Button(oneGifFrame, text = "Go", command = lambda: start1Dgif(int(oneGifEntry.get())))
        oneGifGo.grid(row = 1, column = 0, columnspan = 2, padx = 5, pady = 2.5)

        twoPngFrame = ttk.LabelFrame(startFrame, text = "Two Dimensional PNG")
        twoPngFrame.grid(row = 3, column = 0, padx = 10, pady = 5)

        twoPngText = ttk.Label(twoPngFrame, text = "Generations:")
        twoPngText.grid(row = 0, column = 0, padx = 5, pady = 2.5, sticky = "e")

        global twoPngEntry
        twoPngEntry = ttk.Entry(twoPngFrame)
        twoPngEntry.grid(row = 0, column = 1, padx = 5, pady = 2.5)

        twoPngGo = ttk.Button(twoPngFrame, text = "Go", command = lambda: start2Dpng(int(twoPngEntry.get())))
        twoPngGo.grid(row = 1, column = 0, columnspan = 2, padx = 5, pady = 2.5)

        twoGifFrame = ttk.LabelFrame(startFrame, text = "Two Dimensional GIF")
        twoGifFrame.grid(row = 3, column = 1, padx = 10, pady = 5)

        twoGifText = ttk.Label(twoGifFrame, text = "Generations:")
        twoGifText.grid(row = 0, column = 0, padx = 5, pady = 2.5, sticky = "e")

        global twoGifEntry
        twoGifEntry = ttk.Entry(twoGifFrame)
        twoGifEntry.grid(row = 0, column = 1, padx = 5, pady = 2.5)

        twoGifGo = ttk.Button(twoGifFrame, text = "Go", command = lambda: start2Dgif(int(twoGifEntry.get())))
        twoGifGo.grid(row = 1, column = 0, columnspan = 2, padx = 5, pady = 2.5)

        global progressText
        progressText = ttk.Label(startFrame, wraplength = 450, justify = tk.CENTER)
        progressText.grid(row = 4, column = 0, columnspan = 2)

        global resultsFrame
        resultsFrame = ttk.LabelFrame(startFrame, text = "Results")
        resultsFrame.grid(row = 5, column = 0, columnspan = 2, padx = 10, pady = 5)
        


# General methods
def tryExit():
    response = messagebox.askyesno(title = "Exit?", message = "Are you sure you want to exit?")
    if response:
        root.quit()
    else:
            return

root = Gui()
root.title("Turing Pattern Visualisation")
root.mainloop()