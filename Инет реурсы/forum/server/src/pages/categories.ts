import { prisma } from '$/utils/database' 
import { IncomingMessage, ServerResponse } from 'http'
import ejs from 'ejs'
import fs from 'fs/promises'
import { Category } from '@prisma/client'

export async function categoriesPage(req: IncomingMessage, res: ServerResponse) {
  let categories: Category[] = []
  let error = ''
  let user = null

  try {
    // Получаем информацию о пользователе из сессии
    const cookieHeader = req.headers.cookie
    if (cookieHeader) {
      const cookies = parseCookies(cookieHeader)
      const token = cookies.token || cookies.session
      
      if (token) {
        const session = await prisma.session.findFirst({
          where: {
            token,
            expiresAt: {
              gt: new Date()
            }
          },
          include: {
            user: {
              select: {
                id: true,
                email: true,
                username: true,
                role: true
              }
            }
          }
        })
        
        if (session && session.user) {
          user = session.user
        }
      }
    }

    // Получаем все категории
    categories = await prisma.category.findMany({
      orderBy: {
        name: 'asc',
      },
    })
  } catch (err) {
    console.error('Categories page error:', err)
    error = 'Ошибка при загрузке категорий'
  }

  const template = await fs.readFile('src/pages/templates/categories.ejs', 'utf-8')

  const html = ejs.render(template, {
    heading: 'Все категории',
    categories: categories,
    error: error,
    user: user
  })

  res.writeHead(200, { 
      'Content-Type': 'text/html; charset=utf-8',
      'Content-Length': Buffer.byteLength(html)
    })
  res.end(html)
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