import { prisma } from '$/utils/database'; 
import { IncomingMessage, ServerResponse } from 'http'

export async function categoriesPage(req: IncomingMessage, res: ServerResponse) {
  let categories;
  let error = '';
   try {
    categories = await prisma.category.findMany({
      orderBy: {
        name: 'asc',
      },
    });
  } catch (error) {
    error = error
  };

    
  const html = `
    <!DOCTYPE html>
    <html>
    <head>
        <title>Категории форума</title>
    </head>
    <body>
        <h1>Категории</h1>
        ${error}
        <ul>
            ${categories?.map(cat => `
                <li>
                    <a href="/page/category/${cat.id}">${cat.name}</a>
                </li>
            `).join('')}
        </ul>
    </body>
    </html>
  `;

  res.writeHead(200, { 
      'Content-Type': 'text/html; charset=utf-8',
      'Content-Length': Buffer.byteLength(html)
    });
  res.end(html);
    
  return html
    // sendHtml(res, 200, html);
}