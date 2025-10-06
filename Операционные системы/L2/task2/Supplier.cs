using task2;

internal class Supplier
{
    private CancellationTokenSource cts;
    private Task supplierTask;
    private Queue<string> buffer;
    private Mutex mutex;
    private PictureBox pbox;
    private List<string> imagePaths;
    private int currentDelay = 500;
    private Form1 form;

    public Supplier(Queue<string> buffer, ref Mutex bufferMutex, PictureBox pbox, List<string> imagePaths, Form1 form)
    {
        this.buffer = buffer;
        this.mutex = bufferMutex;
        this.pbox = pbox;
        this.imagePaths = imagePaths;
        this.currentDelay = Form1.delay1;
        this.form = form;
    }

    public void Start()
    {
        cts = new CancellationTokenSource();
        supplierTask = Task.Run(() => DoWork(cts.Token));
    }

    public void UpdateDelay(int newDelay)
    {
        currentDelay = newDelay;
    }

    public void Stop()
    {
        cts?.Cancel();
        try
        {
            supplierTask?.Wait(2000);
        }
        catch (AggregateException) { }
        finally
        {
            cts?.Dispose();
            supplierTask = null;
        }
    }

    private void DoWork(CancellationToken token)
    {
        int currentCount = 0;
        while (!token.IsCancellationRequested)
        {
            try
            {
                // Анимация приготовления ВНЕ мьютекса
                UpdateImageSafe(imagePaths[0]); // Повар готовит
                Thread.Sleep(currentDelay);

                UpdateImageSafe(imagePaths[2]); // Повар с ждет

                // Короткий захват мьютекса для добавления в очередь
                Console.WriteLine($"Supplier хочет в кс");
                mutex.WaitOne();
                try
                {
                    UpdateImageSafe(imagePaths[1]); // Повар с коробкой
                    Console.WriteLine($"Supplier прошел в кс");
                    if (token.IsCancellationRequested) break;
                    buffer.Enqueue($"[{DateTime.Now:T}] pizza");
                    currentCount = buffer.Count;
                    Thread.Sleep(1000);
                    Console.WriteLine($"Supplier добавил пиццу, в очереди: {buffer.Count}");
                    UpdateImageSafe(imagePaths[0]); // Повар готовит
                }
                finally
                {
                    mutex.ReleaseMutex();
                    Console.WriteLine($"Supplier освободил кс");
                }
                UpdateLabelSafe("x" + currentCount.ToString()); // Безопасное обновление

                Thread.Sleep(100);
            }
            catch
            {
                Console.WriteLine($"Supplier error");
            }
        }
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