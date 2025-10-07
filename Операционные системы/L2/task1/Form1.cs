namespace task1
{
    public partial class Form1 : Form
    {
        private TriangleDrawer triangleDrawer;
        private SquareDrawer squareDrawer;
        private string filePath = "points.txt";

        public List<Point[]> drawnShapes = new List<Point[]>();

        public Form1()
        {
            InitializeComponent();
            // Включаем двойную буферизацию для формы
            this.DoubleBuffered = true;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            //Утсанавливаем начальные значения для КС
            SharedData.flagDraw = [false, false];
            SharedData.turnDraw = 0;

            SharedData.flagFile = [false, false];
            SharedData.turnFile = 0;

            SharedData.scroll1 = 250;
            SharedData.scroll2 = 250;

            //Создаем и запускаем потоки
            triangleDrawer = new TriangleDrawer(this, filePath);
            triangleDrawer.Start();

            squareDrawer = new SquareDrawer(this, filePath);
            squareDrawer.Start();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            // Установка высокого качества отрисовки для сглаживания фигур
            e.Graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

            using (Pen bluePen = new Pen(Color.Blue, 2))
            using (Pen redPen = new Pen(Color.Red, 2))
            {
                foreach (var shape in drawnShapes)
                {
                    if (shape.Length == 3) // Треугольник
                    {
                        e.Graphics.DrawPolygon(bluePen, shape);
                    }
                    else if (shape.Length == 4) // Квадрат
                    {
                        e.Graphics.DrawPolygon(redPen, shape);
                    }
                }
            }
        }

        // Добавляем обработчик закрытия формы, чтобы остановить потоки
        private async void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (triangleDrawer != null)
            {
                await triangleDrawer.Stop(); // Теперь ждём async
            }
            if (squareDrawer != null)
            {
                await squareDrawer.Stop();
            }

        }

        private void trackBar1_Scroll(object sender, EventArgs e)
        {
            SharedData.scroll1 = trackBar1.Value;
        }

        private void trackBar2_Scroll(object sender, EventArgs e)
        {
            SharedData.scroll2 = trackBar2.Value;

        }
    }
}