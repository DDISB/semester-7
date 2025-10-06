using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace task1
{
    internal class TriangleDrawer
    {
        private static readonly Random random = new Random();

        private Form1 form;
        private string filePath;
        private StreamReader? fileReader;
        private CancellationTokenSource cts; // Для отмены
        private Task drawingTask; // Сохраняем задачу

        public TriangleDrawer(Form1 form, string filePath)
        {
            this.form = form;
            this.filePath = filePath;
            //cts = new CancellationTokenSource();
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

                // Вход в CS по Петтерсону (myId = 0)
                int myId = 0;
                SharedData.flagDraw[myId] = true;
                SharedData.turnDraw = 1 - myId;
                Console.WriteLine("Треугольник: попытка войти в КС рисовение");
                bool flagConsole = true;
                while (SharedData.flagDraw[1 - myId] && SharedData.turnDraw == (1 - myId))
                {
                    if (flagConsole)
                    {
                        Console.WriteLine("Треугольник: Ожидание доступа к КС");
                        flagConsole = false;
                    }
                    if (token.IsCancellationRequested) break; // Проверка в busy-wait
                    await Task.Yield();
                }

                if (token.IsCancellationRequested)
                {
                    SharedData.flagDraw[myId] = false; // Выход из CS
                    break;
                }

                Console.WriteLine("Треугольник: Работа в КС");
                // Критическая секция: Рисование
                Point point2 = new Point(point.X - 20, point.Y - 20);
                Point point3 = new Point(point.X - 40, point.Y);
                DrawTriangle(point, point2, point3);

                await Task.Delay(SharedData.scroll1, token); // Delay с отменой
                // Выход
                Console.WriteLine("Треугольник: Выход из КС");
                SharedData.flagDraw[myId] = false;

            }
        }

        private Point ReadPointFromFile()
        {
            int myId = 0;
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

        private void DrawTriangle(Point p1, Point p2, Point p3)
        {
            if (form.IsDisposed || form.Disposing) return; // Проверка перед Invoke

            form.Invoke(new Action(() =>
            {
                if (form.IsDisposed || form.Disposing) return; // Ещё раз внутри
                form.drawnShapes.Add(new Point[] { p1, p2, p3 });
                form.Invalidate();
            }));
        }
    }
}