using System.Windows.Forms;
namespace task2
{
    public partial class Form1 : Form
    {
        public static int delay1 = 500;
        public static int delay2 = 500;
        public static int delay3 = 500;
        public Queue<string> buffer = new Queue<string>();
        public List<string> imagePaths = new List<string>();

        public static Mutex bufferMutex = new Mutex();
        private Supplier supplier;
        private Consumer consumer;

        public Form1()
        {
            InitializeComponent();
            InitializeComponents();
        }

        public void SetLabelName(string str)
        {
            label1.Text = str;
        }

        private void Form1_Load(object sender, EventArgs e)
        {

            supplier = new Supplier(buffer, ref bufferMutex, pictureBox1, imagePaths, this);
            supplier.Start();

            consumer = new Consumer(buffer, ref bufferMutex, pictureBox2, imagePaths, this);
            consumer.Start();
        }

        private void InitializeComponents()
        {
            pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox3.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox3.Image = Image.FromFile(@"..\..\..\пицца.png");

            imagePaths.Add(@"..\..\..\повар готовит.png");
            imagePaths.Add(@"..\..\..\повар с коробкой.png");
            imagePaths.Add(@"..\..\..\повар ждет.png");
            imagePaths.Add(@"..\..\..\мики ждет.png");
            imagePaths.Add(@"..\..\..\мики пица.png");
            imagePaths.Add(@"..\..\..\мики коробка.png");

            if (imagePaths.Count > 0 && File.Exists(imagePaths[0]))
            {
                pictureBox1.Image = Image.FromFile(imagePaths[0]);
            }

            if (imagePaths.Count > 0 && File.Exists(imagePaths[3]))
            {
                pictureBox2.Image = Image.FromFile(imagePaths[3]);
            }
        }

        private void trackBar1_Scroll(object sender, EventArgs e)
        {
            delay1 = trackBar1.Value;
            supplier?.UpdateDelay(delay1);
        }

        private void trackBar2_Scroll(object sender, EventArgs e)
        {
            delay2 = trackBar2.Value;
            consumer?.UpdateDelay(delay2);
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            supplier?.Stop();
            consumer?.Stop();
            base.OnFormClosing(e);
        }

        private void trackBar3_Scroll(object sender, EventArgs e)
        {
            delay3 = trackBar3.Value;
            consumer?.UpdateBufferDelay(delay3);
            supplier?.UpdateBufferDelay(delay3);
        }
    }

}