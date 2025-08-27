# Barra

Barra is a package for Linear Regression. For now, its only support Simple Linear Regression (X and Y), i'll update the code for Multi Linear Regression.

<div align="center">
    <img src="/pic/barra-fix.png" alt="Barra Image Formula"/>
</div>

```python

from barra import LinearRegression
import pandas as pd

data = {
    'jam_belajar' : [2,3,5,7],
    'nilai_ujian' : [65,70,80,86]
}

x = data['jam_belajar']
y = data['nilai_ujian']

data_frame = pd.DataFrame(data)
print(data_frame)

ob = LinearRegression(x, y)

print(ob.predict(8))

```

[Code](/barra/linreg.py)
[See on pdf](/pic/Barra%20-%20Linear%20Regression%20From%20Scratch%20-%20Fix.pdf)

