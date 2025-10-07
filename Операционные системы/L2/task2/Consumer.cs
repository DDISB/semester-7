using task2;

internal class Consumer
{
    private CancellationTokenSource cts;
    private Task consumerTask;
    private Queue<string> buffer;
    private Mutex mutex;
    private PictureBox pbox;
    private List<string> imagePaths;
    private int currentDelay = 500;
    private int bufferDelay = 500;
    private Form1 form;

    public Consumer(Queue<string> buffer, ref Mutex bufferMutex, PictureBox pbox, List<string> imagePaths, Form1 form)
    {
        this.buffer = buffer;
        this.mutex = bufferMutex;
        this.pbox = pbox;
        this.imagePaths = imagePaths;
        this.currentDelay = Form1.delay2;
        this.form = form;
    }

    public void Start()
    {
        cts = new CancellationTokenSource();
        consumerTask = Task.Run(() => DoWork(cts.Token));
    }

    public void UpdateDelay(int newDelay)
    {
        currentDelay = newDelay;
    }

    public void UpdateBufferDelay(int newDelay)
    {
        bufferDelay = newDelay;
    }

    public void Stop()
    {
        cts?.Cancel();
        try
        {
            consumerTask?.Wait(2000);
        }
        catch (AggregateException) { }
        finally
        {
            cts?.Dispose();
            consumerTask = null;
        }
    }

    private void DoWork(CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            bool pizzaAvailable = false;
            int currentCount = 0;
            try
            {
                // Короткий захват мьютекса для проверки очереди
                Console.WriteLine($"Consumer хочет в кс");
                mutex.WaitOne();
                try
                {
                    Console.WriteLine($"Consumer прошел в кс");
                    if (token.IsCancellationRequested) break;
                    pizzaAvailable = buffer.Count > 0;
                    if (pizzaAvailable)
                    {
                        UpdateImageSafe(imagePaths[5]); // Мики с пиццей
                        buffer.Dequeue();
                        currentCount = buffer.Count;
                        Thread.Sleep(bufferDelay);
                        Console.WriteLine($"Consumer взял пиццу, в очереди: {buffer.Count}");

                    }
                }
                finally
                {
                    mutex.ReleaseMutex();
                    Console.WriteLine($"Consumer освободил кс");
                }

                if (pizzaAvailable)
                {
                    UpdateImageSafe(imagePaths[4]); // Мики ест

                    UpdateLabelSafe("x" + currentCount.ToString()); // Безопасное обновление
                    // Долгие операции ВНЕ мьютекса
                    Thread.Sleep(currentDelay);
                    UpdateImageSafe(imagePaths[3]); // Мики ждет

                }
                else
                {
                    Thread.Sleep(100);
                }
            }
            catch 
            {
                Console.WriteLine($"Consumer error");
            }
        }

        UpdateImageSafe(imagePaths[2]); // Вернуть в исходное состояние
    }

    private void UpdateImageSafe(string imagePath)
    {
        if (pbox.InvokeRequired)
        {
            pbox.Invoke(new Action<string>(UpdateImageSafe), imagePath);
            return;
        }

        if (File.Exists(imagePath))
        {
            var oldImage = pbox.Image;
            pbox.Image = Image.FromFile(imagePath);
            oldImage?.Dispose();
        }
    }

    private void UpdateLabelSafe(string text)
    {
        if (form.InvokeRequired)
        {
            form.Invoke(new Action<string>(UpdateLabelSafe), text);
            return;
        }

        form.SetLabelName(text);
    }
}