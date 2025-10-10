import { IncomingMessage, ServerResponse } from 'http'
import { prisma } from '../utils/database'

export async function logout(req: IncomingMessage, res: ServerResponse) {
  try {
    const cookieHeader = req.headers.cookie
    let token = null

    // Получаем токен из cookies
    if (cookieHeader) {
      const cookies = parseCookies(cookieHeader)
      token = cookies.token || cookies.session
    }

    // Если есть токен, удаляем сессию из БД
    if (token) {
      const session = await prisma.session.findFirst({
        where: {
          token
        }
      })
      
      if (session) {
        await prisma.session.delete({
          where: {
            id: session.id
          }
        })
        console.log(`Session deleted for user: ${session.userId}`)
      }
    }

    // Устанавливаем cookies с истекшим сроком
    const cookieHeaders = [
      'token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; HttpOnly',
      'session=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; HttpOnly'
    ]

    // Редирект на главную страницу
    res.writeHead(302, {
      'Location': '/page/home',
      'Set-Cookie': cookieHeaders
    })
    res.end()

  } catch (error) {
    console.error('Logout error:', error)
    
    // В случае ошибки все равно делаем редирект и очищаем cookies
    const cookieHeaders = [
      'token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; HttpOnly',
      'session=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; HttpOnly'
    ]

    res.writeHead(302, {
      'Location': '/page/home',
      'Set-Cookie': cookieHeaders
    })
    res.end()
  }
}

// Вспомогательная функция для парсинга cookies
function parseCookies(cookieHeader: string): Record<string, string> {
  const cookies: Record<string, string> = {}
  
  if (!cookieHeader) return cookies
  
  cookieHeader.split(';').forEach(cookie => {
    const [name, value] = cookie.trim().split('=')
    if (name && value) {
      cookies[name] = decodeURIComponent(value)
    }
  })
  
  return cookies
}