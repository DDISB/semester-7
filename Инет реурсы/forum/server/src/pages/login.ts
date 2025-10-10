import { IncomingMessage, ServerResponse } from 'http'
import ejs from 'ejs'
import fs from 'fs/promises'
import { login } from '$/auth/login'

export async function loginPage(req: IncomingMessage, res: ServerResponse) {
  let error = ''

  // Обработка POST запроса
  if (req.method === 'POST') {
    try {
      const body = await getRequestBody(req)
      const formData = new URLSearchParams(body.toString())
      
      const loginData = {
        email: formData.get('email') || '',
        password: formData.get('password') || ''
      }

      const result = await login(loginData)

      if (result.success && result.token) {
        // Успешный вход - устанавливаем куки и редирект
        res.setHeader('Set-Cookie', `token=${result.token} HttpOnly Path=/ Max-Age=${7 * 24 * 60 * 60}`)
        res.writeHead(302, { Location: '/' })
        res.end()
        return
      } else {
        error = result.message || 'Ошибка входа'
      }
    } catch (err) {
      error = 'Ошибка при обработке запроса'
    }
  }

  // Рендер страницы
  const template = await fs.readFile('src/pages/templates/login.ejs', 'utf-8')
  const html = ejs.render(template, { error })

  res.writeHead(200, { 
    'Content-Type': 'text/html charset=utf-8',
    'Content-Length': Buffer.byteLength(html)
  })
  res.end(html)
}

// Вспомогательная функция для получения тела запроса
function getRequestBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', (chunk: Buffer) => chunks.push(chunk))
    req.on('end', () => resolve(Buffer.concat(chunks)))
    req.on('error', reject)
  })
}