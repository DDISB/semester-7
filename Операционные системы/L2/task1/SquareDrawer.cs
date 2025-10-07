//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;

//namespace task1
//{
//    internal class SquareDrawer
//    {
//        private static readonly Random random = new Random();
//        private Form1 form;
//        private bool isRunning = false;
//        private string filePath;
//        private StreamReader? fileReader;

//        public SquareDrawer(Form1 form, string filePath)
//        {
//            this.form = form;
//            this.filePath = filePath;
//            if (File.Exists(filePath))
//            {
//                fileReader = new StreamReader(filePath);
//            }
//        }

//        public void Start()
//        {
//            if (!isRunning)
//            {
//                isRunning = true;
//                Task.Run(DoWork);
//            }
//        }

//        public void Stop()
//        {
//            isRunning = false;
//        }

//        private async void DoWork()
//        {
//            while (isRunning)
//            {
//                Point point = ReadPointFromFile();
//                {
//                    // Вход в CS по Петтерсону (myId = 1)
//                    int myId = 1;
//                    SharedData.flagDraw[myId] = true;
//                    SharedData.turnDraw = 1 - myId; // Уступка очереди другому потоку
//                    while (SharedData.flagDraw[1 - myId] && SharedData.turnDraw == (1 - myId))
//                    {
//                        await Task.Yield(); // Busy-wait с yield для не нагружать CPU
//                    }

//                    // Критическая секция: Рисование
//                    Point point2 = new Point(point.X, point.Y - 20);
//                    Point point3 = new Point(point.X + 40, point.Y - 20);
//                    Point point4 = new Point(point.X + 40, point.Y);
//                    DrawSquare(point, point2, point3, point4);

//                    // Выход
//                    SharedData.flagDraw[myId] = false;
//                }

//                //await Task.Delay(100);
//                await Task.Delay(SharedData.scroll2);
//            }
//        }

//        private Point ReadPointFromFile()
//        {
//            int myId = 1;
//            SharedData.flagFile[myId] = true;
//            SharedData.turnFile = 1 - myId; // Уступка очереди
//            while (SharedData.flagFile[1 - myId] && SharedData.turnFile == (1 - myId))
//            {
//                System.Threading.Thread.Sleep(1);
//            }

//            // Критическая секция: чтение точки из файла
//            Point point = new Point();
//            if (fileReader != null)
//            {
//                string? line = fileReader.ReadLine();
//                if (line == null)
//                {
//                    Stop();
//                }
//                else
//                {
//                    var parts = line.Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
//                    if (parts.Length >= 2)
//                    {
//                        int x = int.Parse(parts[0]);
//                        int y = int.Parse(parts[1]);
//                        point = new Point(x, y);
//                    }
//                }
//            }
//            else
//            {
//                Stop();
//            }

//            // Выход
//            SharedData.flagFile[myId] = false;

//            return point;
//        }

//        private void DrawSquare(Point p1, Point p2, Point p3, Point p4)
//        {
//            form.Invoke(new Action(() =>
//            {
//                form.drawnShapes.Add(new Point[] { p1, p2, p3, p4 });
//                form.Invalidate();
//            }));
//        }
//    }
//}


namespace task1
{
    internal class SquareDrawer
    {
        private static readonly Random random = new Random();

        private Form1 form;
        private string filePath;
        private StreamReader? fileReader;
        private CancellationTokenSource cts; // Для отмены
        private Task drawingTask; // Сохраняем задачу

        public SquareDrawer(Form1 form, string filePath)
        {
            this.form = form;
            this.filePath = filePath;
            cts = new CancellationTokenSource();
            if (File.Exists(filePath))
            {
                fileReader = new StreamReader(filePath);
            }
        }

        public void Start()
        {
            if (drawingTask == null || drawingTask.IsCompleted)
            {
                cts = new CancellationTokenSource(); // Новый CTS на каждый запуск
                drawingTask = Task.Run(() => DoWork(cts.Token));
            }
        }

        public async Task Stop() // Теперь async Task, чтобы ждать
        {
            cts.Cancel(); // Отмена
            try
            {
                if (drawingTask != null)
                {
                    await drawingTask; // Ждём завершения задачи
                }
            }
            catch (TaskCanceledException) { } // Игнорируем отмену
            finally
            {
                fileReader?.Close();
                fileReader?.Dispose();
                fileReader = null;
            }
        }

        private async Task DoWork(CancellationToken token) // Теперь async Task
        {
            while (!token.IsCancellationRequested) // Вместо isRunning
            {
                Point point = ReadPointFromFile();
                if (token.IsCancellationRequested) break; // Проверка после чтения

                // Вход в CS по Петтерсону (myId = 1)
                int myId = 1;
                SharedData.flagDraw[myId] = true;
                SharedData.turnDraw = 1 - myId;
                Console.WriteLine("Квадрат: попытка войти в КС рисовение");
                bool flagConsole = true;

                while (SharedData.flagDraw[1 - myId] && SharedData.turnDraw == (1 - myId))
                {
                    if (flagConsole)
                    {
                        flagConsole = false;
                        Console.WriteLine("Квадрат: Ожидание доступа к КС");
                    }
                    if (token.IsCancellationRequested) break; // Проверка в busy-wait
                    await Task.Yield();
                }

                if (token.IsCancellationRequested)
                {
                    SharedData.flagDraw[myId] = false; // Выход из CS
                    break;
                }
                Console.WriteLine("Квадрат: Работа в КС");
                // Критическая секция: Рисование
                Point point2 = new Point(point.X, point.Y - 20);
                Point point3 = new Point(point.X + 40, point.Y - 20);
                Point point4 = new Point(point.X + 40, point.Y);
                DrawSquare(point, point2, point3, point4);

                // Выход
                Console.WriteLine("Квадрат: Выход из КС");
                SharedData.flagDraw[myId] = false;
                await Task.Delay(SharedData.scroll2, token); // Delay с отменой
            }
        }

        private Point ReadPointFromFile()
        {
            int myId = 1;
            SharedData.flagFile[myId] = true;
            SharedData.turnFile = 1 - myId;
            while (SharedData.flagFile[1 - myId] && SharedData.turnFile == (1 - myId))
            {
                Thread.Sleep(1);
            }

            // Критическая секция: чтение
            Point point = new Point();
            if (fileReader != null)
            {
                string? line = fileReader.ReadLine();
                if (line == null)
                {
                    // Stop() теперь вызывается извне, здесь просто завершаем
                }
                else
                {
                    var parts = line.Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2)
                    {
                        int x = int.Parse(parts[0]);
                        int y = int.Parse(parts[1]);
                        point = new Point(x, y);
                    }
                }
            }

            SharedData.flagFile[myId] = false;
            return point;
        }


        private void DrawSquare(Point p1, Point p2, Point p3, Point p4)
        {
            if (form.IsDisposed || form.Disposing) return; // Проверка перед Invoke

            form.Invoke(new Action(() =>
            {
                if (form.IsDisposed || form.Disposing) return; // Ещё раз внутри

                form.drawnShapes.Add(new Point[] { p1, p2, p3, p4 });
                form.Invalidate();
            }));
        }
    }
}