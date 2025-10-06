namespace task1
{
    partial class settingForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            button1 = new Button();
            trackBar1 = new TrackBar();
            trackBar2 = new TrackBar();
            label1 = new Label();
            label2 = new Label();
            ((System.ComponentModel.ISupportInitialize)trackBar1).BeginInit();
            ((System.ComponentModel.ISupportInitialize)trackBar2).BeginInit();
            SuspendLayout();
            // 
            // button1
            // 
            button1.Location = new Point(258, 218);
            button1.Name = "button1";
            button1.Size = new Size(249, 84);
            button1.TabIndex = 0;
            button1.Text = "Запуск";
            button1.UseVisualStyleBackColor = true;
            button1.Click += button1_Click;
            // 
            // trackBar1
            // 
            trackBar1.Location = new Point(92, 97);
            trackBar1.Maximum = 1000;
            trackBar1.Minimum = 10;
            trackBar1.Name = "trackBar1";
            trackBar1.Size = new Size(104, 45);
            trackBar1.TabIndex = 1;
            trackBar1.Value = 100;
            trackBar1.Scroll += trackBar1_Scroll;
            // 
            // trackBar2
            // 
            trackBar2.Location = new Point(558, 97);
            trackBar2.Maximum = 1000;
            trackBar2.Minimum = 10;
            trackBar2.Name = "trackBar2";
            trackBar2.Size = new Size(104, 45);
            trackBar2.TabIndex = 2;
            trackBar2.Value = 100;
            trackBar2.Scroll += trackBar2_Scroll;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(92, 46);
            label1.Name = "label1";
            label1.Size = new Size(145, 15);
            label1.TabIndex = 3;
            label1.Text = "Задержка треугольников";
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new Point(558, 46);
            label2.Name = "label2";
            label2.Size = new Size(118, 15);
            label2.TabIndex = 4;
            label2.Text = "Задержка квадратов";
            // 
            // settingForm
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(800, 450);
            Controls.Add(label2);
            Controls.Add(label1);
            Controls.Add(trackBar2);
            Controls.Add(trackBar1);
            Controls.Add(button1);
            Name = "settingForm";
            Text = "settingForm";
            ((System.ComponentModel.ISupportInitialize)trackBar1).EndInit();
            ((System.ComponentModel.ISupportInitialize)trackBar2).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button button1;
        private TrackBar trackBar1;
        private TrackBar trackBar2;
        private Label label1;
        private Label label2;
    }
}