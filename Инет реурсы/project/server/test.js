const http = require('http');
const url = require('url');

// Данные для генерации случайной информации
const randomData = {
  animals: ['🐱 Кот', '🐶 Собака', '🐰 Кролик', '🦊 Лиса', '🐻 Медведь', '🐼 Панда'],
  colors: ['🔴 Красный', '🔵 Синий', '🟢 Зеленый', '🟡 Желтый', '🟣 Фиолетовый', '🟠 Оранжевый'],
  cities: ['🏙️ Москва', '🌆 Нью-Йорк', '🏰 Париж', '🗼 Токио', '🌃 Лондон', '🏔️ Сидней'],
  quotes: [
    'Жизнь — это то, что происходит, пока ты строишь планы.',
    'Всё, что ты можешь вообразить — реально.',
    'Дорогу осилит идущий.',
    'Лучше поздно, чем никогда.'
  ],
  facts: [
    'Коты спят около 70% своей жизни.',
    'Мед — единственная пища, которая не портится.',
    'Сердце креветки находится в её голове.',
    'Страусы бегают быстрее лошадей.'
  ]
};

// Функция для получения случайного элемента из массива
function getRandomItem(array) {
  return array[Math.floor(Math.random() * array.length)];
}

// Функция для генерации случайной информации
function generateRandomInfo() {
  const types = ['animal', 'color', 'city', 'quote', 'fact', 'mixed'];
  const type = getRandomItem(types);
  
  switch (type) {
    case 'animal':
      return { type: 'Животное', data: getRandomItem(randomData.animals) };
    case 'color':
      return { type: 'Цвет', data: getRandomItem(randomData.colors) };
    case 'city':
      return { type: 'Город', data: getRandomItem(randomData.cities) };
    case 'quote':
      return { type: 'Цитата', data: getRandomItem(randomData.quotes) };
    case 'fact':
      return { type: 'Факт', data: getRandomItem(randomData.facts) };
    case 'mixed':
      return {
        type: 'Микс',
        data: {
          animal: getRandomItem(randomData.animals),
          color: getRandomItem(randomData.colors),
          fact: getRandomItem(randomData.facts)
        }
      };
  }
}

// Создаем HTTP сервер
const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;
  const method = req.method;

  // Устанавливаем заголовки для CORS (чтобы можно было тестировать из браузера)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Обработка preflight запросов (для CORS)
  if (method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // Главная страница
  if (pathname === '/' && method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>Случайный сервер</title>
        <meta charset="utf-8">
        <style>
          body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
          .info { background: #f0f0f0; padding: 20px; border-radius: 10px; margin: 20px 0; }
          button { padding: 10px 20px; margin: 5px; cursor: pointer; }
          pre { background: #eee; padding: 10px; border-radius: 5px; }
        </style>
      </head>
      <body>
        <h1>🎲 Сервер случайной информации</h1>
        <p>Используйте:</p>
        <ul>
          <li><strong>GET /random</strong> - получить случайную информацию</li>
          <li><strong>POST /random</strong> - получить расширенную случайную информацию</li>
        </ul>
        <div>
          <button onclick="fetchRandom('GET')">Тест GET запроса</button>
          <button onclick="fetchRandom('POST')">Тест POST запроса</button>
        </div>
        <div id="result"></div>
        
        <script>
          async function fetchRandom(method) {
            const options = method === 'POST' ? { 
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ timestamp: Date.now() })
            } : {};
            
            try {
              const response = await fetch('/random', options);
              const data = await response.json();
              document.getElementById('result').innerHTML = 
                '<div class="info"><h3>' + data.type + '</h3><pre>' + 
                JSON.stringify(data.data, null, 2) + '</pre><p>Время: ' + 
                new Date(data.timestamp).toLocaleString() + '</p></div>';
            } catch (error) {
              console.error('Ошибка:', error);
            }
          }
        </script>
      </body>
      </html>
    `);
    return;
  }

  // Обработка запросов к /random
  if (pathname === '/random') {
    if (method === 'GET') {
      // Простая случайная информация для GET
      const randomInfo = generateRandomInfo();
      
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({
        message: 'GET запрос обработан успешно!',
        type: randomInfo.type,
        data: randomInfo.data,
        timestamp: new Date().toISOString(),
        method: 'GET'
      }, null, 2));
      
    } else if (method === 'POST') {
      // Собираем тело POST запроса
      let body = '';
      
      req.on('data', chunk => {
        body += chunk.toString();
      });
      
      req.on('end', () => {
        let postData = {};
        try {
          postData = JSON.parse(body || '{}');
        } catch (e) {
          postData = { error: 'Невалидный JSON' };
        }
        
        // Расширенная случайная информация для POST
        const randomInfos = [];
        for (let i = 0; i < 3; i++) {
          randomInfos.push(generateRandomInfo());
        }
        
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({
          message: 'POST запрос обработан успешно!',
          receivedData: postData,
          items: randomInfos,
          timestamp: new Date().toISOString(),
          method: 'POST',
          count: randomInfos.length
        }, null, 2));
      });
      
    } else {
      // Метод не поддерживается
      res.writeHead(405, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({
        error: 'Метод не поддерживается',
        allowedMethods: ['GET', 'POST']
      }));
    }
    return;
  }

  // Обработка несуществующих путей
  res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify({
    error: 'Страница не найдена',
    availableEndpoints: ['GET /', 'GET /random', 'POST /random']
  }));
});

// Запускаем сервер на порту 3000
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`🎯 Сервер запущен на порту ${PORT}`);
  console.log(`📍 Откройте в браузере: http://localhost:${PORT}`);
  console.log(`🔗 Доступные endpoints:`);
  console.log(`   GET  / - Главная страница с тестовым интерфейсом`);
  console.log(`   GET  /random - Получить одну случайную запись`);
  console.log(`   POST /random - Получить несколько случайных записей`);
});

// Обработка graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Остановка сервера...');
  server.close(() => {
    console.log('✅ Сервер остановлен');
    process.exit(0);
  });
});