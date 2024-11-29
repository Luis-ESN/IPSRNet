# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:38:39 2024

@author: Luís Eduardo Sales do Nascimento
"""

from ACE import ACE
from STEREO import STEREO
import IPSRNet

import warnings
warnings.filterwarnings('ignore')

modelo20 = IPSRNet.IPSR20N()
modelo30 = IPSRNet.IPSR30N()
modelo60 = IPSRNet.IPSR60N()
modelo120 = IPSRNet.IPSR120N()

s = '2015-03-17 04:05:00'

name = 'Case studies/ACE 20 ' + s.replace(':', '-') + '.png'
ACE(s, name, time_window=10)
saida = modelo30.predict(name)
print('FF shock at 2015-03-17 04:05')
print('20 minutes Net -->', saida)

name = 'Case studies/ACE 30 ' + s.replace(':', '-') + '.png'
ACE(s, name, time_window=15)
saida = modelo30.predict(name)
print('30 minutes Net -->', saida)

name = 'Case studies/ACE 60 ' + s.replace(':', '-') + '.png'
ACE(s, name, time_window=30)
saida = modelo30.predict(name)
print('60 minutes Net -->', saida)

name = 'Case studies/ACE 120 ' + s.replace(':', '-') + '.png'
ACE(s, name, time_window=60)
saida = modelo30.predict(name)
print('120 minutes Net -->', saida)

print('\n\n')

s = '2016-07-22 23:12:00'

name = 'Case studies/STEREO-A 20 ' + s.replace(':', '-') + '.png'
STEREO(s, name, time_window=10, spacecraft='STA')
saida = modelo30.predict(name)
print('FR shock at 2016-07-22 23:12')
print('20 minutes Net -->', saida)

name = 'Case studies/STEREO-A 30 ' + s.replace(':', '-') + '.png'
STEREO(s, name, time_window=15, spacecraft='STA')
saida = modelo30.predict(name)
print('30 minutes Net -->', saida)

name = 'Case studies/STEREO-A 60 ' + s.replace(':', '-') + '.png'
STEREO(s, name, time_window=30, spacecraft='STA')
saida = modelo30.predict(name)
print('60 minutes Net -->', saida)

name = 'Case studies/STEREO-A 120 ' + s.replace(':', '-') + '.png'
STEREO(s, name, time_window=60, spacecraft='STA')
saida = modelo30.predict(name)
print('120 minutes Net -->', saida)