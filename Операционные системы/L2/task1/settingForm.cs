using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.DataFormats;

namespace task1
{

    public partial class settingForm : Form
    {
        public settingForm()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Form1 newForm = new Form1();  // Создаём экземпляр новой формы (замените Form2 на имя вашей формы)
            newForm.ShowDialog();
        }

       
    }
}
