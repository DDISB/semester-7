# read_table.py

def read_table(filename):
    try:
        from simpledbf import Dbf5
        import numpy as np
        
        # Чтение DBF файла
        dbf = Dbf5(filename)
        df = dbf.to_dataframe()
        
        print(f"Файл успешно прочитан. Столбцы: {list(df.columns)}")
        print(f"Размер данных: {df.shape}")
        print("\nПервые 5 строк данных:")
        print(df.head())
        
        # Правильное извлечение признаков и целевой переменной
        X = df.drop('RATING', axis=1).values  # все столбцы кроме RATING
        y = df['RATING'].values  # только столбец RATING
        
        # Преобразование y в массив массивов чисел
        y_reshaped = y.reshape(-1, 1)  # преобразуем в 2D массив с одним столбцом
        
        print(f"\nПризнаки (X) shape: {X.shape}")
        print(f"Целевая переменная (y) shape: {y_reshaped.shape}")
        
        return X, y_reshaped
        
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return None, None