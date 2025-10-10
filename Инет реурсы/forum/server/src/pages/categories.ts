import { prisma } from '$/utils/database' 
import { IncomingMessage, ServerResponse } from 'http'
import ejs from 'ejs'
import fs from 'fs/promises'

export async function categoriesPage(req: IncomingMessage, res: ServerResponse) {
  let categories
  let error = ''
   try {
    categories = await prisma.category.findMany({
      orderBy: {
        name: 'asc',
      },
    })
  } catch (error) {
    error = error
  }

  const template = await fs.readFile('src/pages/templates/categories.ejs', 'utf-8')

  const html = ejs.render(template, {
    heading: 'Все категории',
    categories: categories,
    error: error
  })

  res.writeHead(200, { 
      'Content-Type': 'text/html charset=utf-8',
      'Content-Length': Buffer.byteLength(html)
    })
  res.end(html)
    
  return html
    // sendHtml(res, 200, html)
}