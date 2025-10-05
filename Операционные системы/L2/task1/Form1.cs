using System;

namespace task1
{
    public static class SharedData
    {
        // Общие данные, к которым нужен доступ (например, буфер или холст)
        public static volatile bool[] flagDraw = new bool[2] { false, false };
        public static volatile int turnDraw = 0;

        public static volatile bool[] flagFile = new bool[2] { false, false };
        public static volatile int turnFile = 0;
    }

    internal class TriangleDrawer
    {
        private static readonly Random random = new Random();

        private Form1 form;
        private bool isRunning = false;
        private string filePath;
        private StreamReader? fileReader;

        public TriangleDrawer(Form1 form, string filePath)
        {
            this.form = form;
            this.filePath = filePath;
            if (File.Exists(filePath))
            {
                fileReader = new StreamReader(filePath);
            }
        }

        public void Start()
        {
            if (!isRunning)
            {
                isRunning = true;
                Task.Run(DoWork);
            }
        }

        public void Stop()
        {
            isRunning = false;
        }

        private async void DoWork()
        {
            while (isRunning)
            {
                Point point = ReadPointFromFile();
                {
                    // Вход в CS по Петтерсону (myId = 0)
                    int myId = 0;
                    SharedData.flagDraw[myId] = true;
                    SharedData.turnDraw = myId;
                    while (SharedData.flagDraw[1 - myId] && SharedData.turnDraw == myId)
                    {
                        await Task.Yield(); // Busy-wait с yield для не нагружать CPU
                    }

                    // Критическая секция: Рисование
                    Point point2 = new Point(point.X - 20, point.Y - 20);
                    Point point3 = new Point(point.X - 40, point.Y);
                    DrawTriangle(point, point2, point3);

                    // Выход
                    SharedData.flagDraw[myId] = false;
                }

                int randomDelay = random.Next(300, 801); 
                await Task.Delay(randomDelay);
            }
        }

        private Point ReadPointFromFile()
        {
            int myId = 0;
            SharedData.flagFile[myId] = true;
            SharedData.turnFile = myId;
            while (SharedData.flagFile[1 - myId] && SharedData.turnFile == myId)
            {
                System.Threading.Thread.Sleep(1);
            }

            // Критическая секция: чтение точки из файла
            Point point = new Point();
            if (fileReader != null)
            {
                string? line = fileReader.ReadLine();
                if (line == null)
                {
                    Stop();
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
            else
            {
                Stop();
            }

            // Выход
            SharedData.flagFile[myId] = false;

            return point;
        }

        private void DrawTriangle(Point p1, Point p2, Point p3)
        {
            form.Invoke(new Action(() =>
            {
                form.drawnShapes.Add(new Point[] { p1, p2, p3 });
                form.Invalidate();
            }));
        }
    }

    internal class SquareDrawer
    {
        private static readonly Random random = new Random();
        private Form1 form;
        private bool isRunning = false;
        private string filePath;
        private StreamReader? fileReader;

        public SquareDrawer(Form1 form, string filePath)
        {
            this.form = form;
            this.filePath = filePath;
            if (File.Exists(filePath))
            {
                fileReader = new StreamReader(filePath);
            }
        }

        public void Start()
        {
            if (!isRunning)
            {
                isRunning = true;
                Task.Run(DoWork);
            }
        }

        public void Stop()
        {
            isRunning = false;
        }

        private async void DoWork()
        {
            while (isRunning)
            {
                Point point = ReadPointFromFile();
                {
                    // Вход в CS по Петтерсону (myId = 0)
                    int myId = 1;
                    SharedData.flagDraw[myId] = true;
                    SharedData.turnDraw = myId;
                    while (SharedData.flagDraw[1 - myId] && SharedData.turnDraw == myId)
                    {
                        await Task.Yield(); // Busy-wait с yield для не нагружать CPU
                    }

                    // Критическая секция: Рисование
                    Point point2 = new Point(point.X, point.Y - 20);
                    Point point3 = new Point(point.X + 40, point.Y - 20);
                    Point point4 = new Point(point.X + 40, point.Y);
                    DrawSquare(point, point2, point3, point4);

                    // Выход
                    SharedData.flagDraw[myId] = false;
                }

                //int randomDelay = random.Next(300, 801);
                //await Task.Delay(randomDelay);
                await Task.Delay(100);
            }
        }

        private Point ReadPointFromFile()
        {
            int myId = 1;
            SharedData.flagFile[myId] = true;
            SharedData.turnFile = myId;
            while (SharedData.flagFile[1 - myId] && SharedData.turnFile == myId)
            {
                System.Threading.Thread.Sleep(1);
            }

            // Критическая секция: чтение точки из файла
            Point point = new Point();
            if (fileReader != null)
            {
                string? line = fileReader.ReadLine();
                if (line == null)
                {
                    Stop();
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
            else
            {
                Stop();
            }

            // Выход
            SharedData.flagFile[myId] = false;

            return point;
        }

        private void DrawSquare(Point p1, Point p2, Point p3, Point p4)
        {
            form.Invoke(new Action(() =>
            {
                form.drawnShapes.Add(new Point[] { p1, p2, p3, p4 });
                form.Invalidate();
            }));
        }
    }
    public partial class Form1 : Form
    {
        private TriangleDrawer triangleDrawer;
        private SquareDrawer squareDrawer;
        private string filePath = "points.txt";

        public List<Point[]> drawnShapes = new List<Point[]>();

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            triangleDrawer = new TriangleDrawer(this, filePath);
            triangleDrawer.Start();

            squareDrawer = new SquareDrawer(this, filePath);
            squareDrawer.Start();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            using (Graphics g = e.Graphics)
            {
                foreach (var shape in drawnShapes)
                {
                    g.DrawPolygon(Pens.Blue, shape);
                }
            }
        }
    }
}