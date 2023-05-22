import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from IPython.display import Audio, display #přehrávání a načítání audia v mp3 formátu
import os #knihovna pro zajištění cesty k audio souboru
import scipy.signal
from scipy.fft import fft
from pydub import AudioSegment
from matplotlib.animation import FuncAnimation
import subprocess
#možná bude třeba nainstalovat: sudo apt install ffmpeg

#prvotni inicializace nazvu skladby i s formatem a nacteni cesty
file_name = "Liszt_liebestraum3(Lisitsa).mp3"
file_path = os.path.join(os.getcwd(), file_name)


def play_audio(file_path):
    audio = Audio(file_path) 
    return display(audio) #zobrazení play lišty
"""funkce vytvoří play lištu a umožní přehrání audia - de facto kontrola správnosti načtení audia"""


def librosa_load (file_name):
    #konverze mp3 formátu do formátu wav, který je podporován při načítání do řetězce
    #Jelikož se v projektu nebudeme zvlášť zabývat konverzí, můžeme ji nechat jako součást této funkce.
    input_file = file_name
    output_file = "output.wav"

    sound = AudioSegment.from_mp3(input_file) #načtení souboru MP3
    sound.export(output_file, format="wav") #uložení souboru WAV
    output_file_path = os.path.join(os.getcwd(), output_file) 

    """načtení audia jako pole hodnot"""
    y, sr = librosa.load(output_file_path) #sr=sample rate, vzorkovací frekvence, y=jednotlivé diskrétní hodnoty
    return y, sr
    """cílem této funkce bylo převézt audio na diskrétní signál (můžeme jej nazvat také jako digitální), který jsme vyjádřili jako konečnou posloupnost hodnot na intervalu (čas skladby)"""
    #sr(vzorkovací frekvence) je zde nastavena na polovinu běžné vzorkovací frekvence (44100 Hz), což stačí pro účely práce se zvukem např. klavíru,
    #jehož maximální frekvence je cca 4200 Hz (hodnota pětičárkového c pro klasické ladění, nebereme v úvahu vyšší harmonické tóny), takže dojde k zachování Shannonova teorému.



def sonograph (y):
    D = librosa.stft(y)
    db = librosa.amplitude_to_db(D, ref=np.max)
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(db, x_axis='time', y_axis='log')

    ax.set(title='Spectrogram')
    #fig.savefig('spektrogram.jpg', dpi=300, bbox_inches='tight')
    #fig.savefig ulozi spektogram jako jpg obrazek - to není nutným krokem
    spectrogram = np.abs(D)
    spectrogram_shape = spectrogram.shape
    return fig, ax, spectrogram_shape
    """funkce vykreslí sonograf z digitálního signálu. Celkově vrací fig, ax a spectrogram_shape, které jsou použity při "animaci". """



#třída pro vykreslování svislé čáry jako ukazatele
class Ukazatel:
    def __init__(self, x):
        self.pozice = x
        self.vykresleno = None
    
    def vykresleni(self, x):
        if self.vykresleno: self.vykresleno.remove()
        self.pozice=x
        self.vykresleno = plt.axvline(x=self.pozice, color="red")
        return self.vykresleno
    """funkce vykresleni vykresli vzdy prave jeden ukazatel na pozici "x", a drive vytvoreny zrusi"""

    def posun(self, x):
        if self.vykresleno: self.vykresleno.remove()
        self.pozice += x
        self.vykresleno = plt.axvline(x=self.pozice, color="red")
        return self.vykresleno    
    """funkce posun vykresli prave jednu svislou caru posunutou o "x" oproti puvodnimu ukazateli"""



def animovani(audio_data, sr):
    fig, ax, spectrogram_shape = sonograph(audio_data)
    num_columns = spectrogram_shape[1]  # počet časových okamžiků v sonografu/spektrogramu
    objekt = Ukazatel(0)
    frame_numbers = int(int(librosa.get_duration(y=audio_data, sr=sr))) 
    #ukazatel se vykresli kazdou vterinu, proto frame_numbers = pocet vterin audia

    #inicializace animace
    def init():
        objekt.vykresleni(0)
        return ax.lines

    #samotna animace prvku - nastaveni posunu/vykresleni v kazde vterine
    def animate(i):
        time_interval = i
        position = time_interval
        objekt.vykresleni(position)
        return ax.lines
    """funkce animate v kazdem framu vykresli ukazatel na pozici "i" vterin"""

    anim = FuncAnimation(fig, animate, init_func=init, frames=frame_numbers, interval=1000, blit=True) #interval je uveden v milisekundach
    anim.save("audio.mp4")
    """funkci FuncAnimation jsou dány argumenty tak, aby byla vytvořena animace kurzoru každou vteřinu posouvající se adekvátně s časem skladby"""
    
    #spojeni audia a videa
    audio = file_name #nazev skladby, uveden na zacatku
    video = 'audio.mp4'

    output = 'vysledek.mp4'

    command = ['ffmpeg', '-i', video, '-i', audio, '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', output]
    vystup=subprocess.call(command)

    """funkce call z knihovny subprocess zavolá nainicializovaný command na spojení audia a videa v jeden celek formátu mp4 
    prostřednictvím ffmpeg (Fast Forward Moving Picture Experts Group)"""
    
    return vystup

