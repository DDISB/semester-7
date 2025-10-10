import { prisma } from '$/utils/database' 
import { IncomingMessage, ServerResponse } from 'http'
import ejs from 'ejs'
import fs from 'fs/promises'

export async function homePage(req: IncomingMessage, res: ServerResponse) {

  const template = await fs.readFile('src/pages/templates/home.ejs', 'utf-8')

  const html = ejs.render(template, {
  })

  res.writeHead(200, { 
      'Content-Type': 'text/html charset=utf-8',
      'Content-Length': Buffer.byteLength(html)
    })
  res.end(html)
    
  return html
}