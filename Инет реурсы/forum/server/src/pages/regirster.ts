import { IncomingMessage, ServerResponse } from 'http'
import ejs from 'ejs';
import fs from 'fs/promises';
import { register } from '$/auth/register';

export async function registerPage(req: IncomingMessage, res: ServerResponse) {
  let error = '';
  let success = '';

  // Обработка GET запроса - показ формы
  if (req.method === 'GET') {
    const template = await fs.readFile('src/pages/templates/register.ejs', 'utf-8');
    const html = ejs.render(template, { error, success });

    res.writeHead(200, { 
      'Content-Type': 'text/html; charset=utf-8',
      'Content-Length': Buffer.byteLength(html)
    });
    res.end(html);
    return;
  }

  // Обработка POST запроса - регистрация
  if (req.method === 'POST') {
    try {
      const body = await getRequestBody(req);
      const formData = new URLSearchParams(body.toString());
      
      const registerData = {
        email: formData.get('email') || '',
        username: formData.get('username') || '',
        password: formData.get('password') || ''
      };

      // Базовая валидация
      if (!registerData.email || !registerData.username || !registerData.password) {
        error = 'Все поля обязательны для заполнения';
      } else if (registerData.password.length < 6) {
        error = 'Пароль должен быть не менее 6 символов';
      } else {
        const result = await register(registerData);

        if (result.success && result.token) {
          // Успешная регистрация - устанавливаем куки
          res.setHeader('Set-Cookie', `token=${result.token}; HttpOnly; Path=/; Max-Age=${7 * 24 * 60 * 60}`);
          
          // Показываем страницу с сообщением об успехе
          success = result.message || 'Регистрация успешна!';
          const template = await fs.readFile('src/pages/templates/register.ejs', 'utf-8');
          const html = ejs.render(template, { error: '', success });

          res.writeHead(200, { 
            'Content-Type': 'text/html; charset=utf-8',
            'Content-Length': Buffer.byteLength(html)
          });
          res.end(html);
          return;
        } else {
          error = result.message || 'Ошибка регистрации';
        }
      }
    } catch (err) {
      console.error('Registration page error:', err);
      error = 'Ошибка при обработке запроса';
    }

    // Если есть ошибка, показываем форму с ошибкой
    const template = await fs.readFile('src/pages/templates/register.ejs', 'utf-8');
    const html = ejs.render(template, { error, success: '' });

    res.writeHead(200, { 
      'Content-Type': 'text/html; charset=utf-8',
      'Content-Length': Buffer.byteLength(html)
    });
    res.end(html);
  }
}

// Вспомогательная функция для получения тела запроса
function getRequestBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}