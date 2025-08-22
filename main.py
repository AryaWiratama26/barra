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
print(x)
print(y)


ob = LinearRegression(x, y)

print(ob.predict(8))
