// import { IncomingMessage, ServerResponse } from 'http'
// import { Subcategory } from '@prisma/client';
// import ejs from 'ejs';
// import fs from 'fs/promises';
// import { prisma } from '../utils/database';

// export async function categoryPage(req: IncomingMessage, res: ServerResponse, categoryId: string) {
//   let category = null;
//   let subcategories: Subcategory[] = [];
//   let error = '';

//   try {
//     // Получаем категорию по ID
//     category = await prisma.category.findUnique({
//       where: {
//         id: categoryId
//       }
//     });

//     if (!category) {
//       error = 'Категория не найдена';
//     } else {
//       // Получаем подкатегории для этой категории
//       subcategories = await prisma.subcategory.findMany({
//       where: {
//         categoryId
//       },
//       orderBy: {
//         name: 'asc',
//       },
//     })
//     }
//   } catch (err) {
//     console.error('Category page error:', err);
//     error = 'Ошибка при загрузке категории';
//   }

//   // Читаем и рендерим шаблон
//   const template = await fs.readFile('src/pages/templates/category.ejs', 'utf-8');
  
//   const html = ejs.render(template, {
//     category: category || { name: 'Не найдена' },
//     subcategories,
//     error
//   });

//   res.writeHead(200, { 
//     'Content-Type': 'text/html; charset=utf-8',
//     'Content-Length': Buffer.byteLength(html)
//   });
//   res.end(html);
// }


import { IncomingMessage, ServerResponse } from 'http'
import { Subcategory } from '@prisma/client';
import ejs from 'ejs';
import fs from 'fs/promises';
import { prisma } from '../utils/database';

export async function categoryPage(req: IncomingMessage, res: ServerResponse, categoryId: string) {
  let category = null;
  let subcategories: Subcategory[] = [];
  let error = '';
  let user = null;

  try {
    // Получаем информацию о пользователе из сессии
    const cookieHeader = req.headers.cookie;
    if (cookieHeader) {
      const cookies = parseCookies(cookieHeader);
      const token = cookies.token || cookies.session;
      
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
        });
        
        if (session && session.user) {
          user = session.user;
        }
      }
    }

    // Получаем категорию по ID
    category = await prisma.category.findUnique({
      where: {
        id: categoryId
      }
    });

    if (!category) {
      error = 'Категория не найдена';
    } else {
      // Получаем подкатегории для этой категории
      subcategories = await prisma.subcategory.findMany({
        where: {
          categoryId
        },
        orderBy: {
          name: 'asc',
        },
      });
    }
  } catch (err) {
    console.error('Category page error:', err);
    error = 'Ошибка при загрузке категории';
  }

  // Читаем и рендерим шаблон
  const template = await fs.readFile('src/pages/templates/category.ejs', 'utf-8');
  
  const html = ejs.render(template, {
    category: category || { name: 'Не найдена', id: '' },
    subcategories,
    error,
    user // Передаем информацию о пользователе в шаблон
  });

  res.writeHead(200, { 
    'Content-Type': 'text/html; charset=utf-8',
    'Content-Length': Buffer.byteLength(html)
  });
  res.end(html);
}

// Вспомогательная функция для парсинга cookies
function parseCookies(cookieHeader: string): Record<string, string> {
  const cookies: Record<string, string> = {};
  
  if (!cookieHeader) return cookies;
  
  cookieHeader.split(';').forEach(cookie => {
    const [name, value] = cookie.trim().split('=');
    if (name && value) {
      cookies[name] = decodeURIComponent(value);
    }
  });
  
  return cookies;
}