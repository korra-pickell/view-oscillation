import time, os
import pyautogui as pg

folder = r'E:\DATA\View-Oscillation-5k'

pg.PAUSE = 0.8 

for i in range(23,46):

    # Output Menu
    pg.click(1295,505)
    pg.click(1295,505)
    pg.click(1295,505)

    # Output Filename
    pg.moveTo(1800,873,0.5)
    pg.click(1800,873)
    pg.hotkey('delete')
    pg.write(os.path.join(folder,str(i)+'\\'))

    # Cam Rotation Property
    pg.click(1295,660)
    pg.click(1295,660)
    pg.click(1295,660)
    time.sleep(0.1)

    # Degree Rotation Input
    pg.click(1730,640)
    pg.click(1730,640)
    pg.hotkey('del')
    pg.write(str(i))

    # Select Render View
    pg.moveTo(1150,512)
    pg.click(1150,525)

    # Start Render
    pg.hotkey('ctrl','f12',interval=0.25)
    s=input('..')
    time.sleep(85)
    
    # Close Render Window
    pg.moveTo(1400,225,0.2)
    pg.click(1400,225)
    time.sleep(1)

    # Ensure Circle Object is Reselected for Rotation
    pg.moveTo(1600,125)
    pg.click(1600,125)
    pg.click(1600,125)
    pg.click(1600,125)

