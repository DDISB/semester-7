import http from 'http'
import { URL } from 'url'
import { IncomingMessage, ServerResponse } from 'http'
import { getRequestBody } from '$/utils/getRequestBody'
import { getSession } from '$/utils/auth'
import { categoriesPage } from './categories'
import { homePage } from './home'
import { loginPage } from './login'
import { registerPage } from './regirster'

export async function handlePage(
  req: IncomingMessage,
  res: ServerResponse,
  method: string | undefined,
  pathname: string) {

    if (method === 'GET' && pathname === '/page/home') {
      await homePage(req, res)
    } else if (method === 'GET' && pathname === '/page/login') {
      await loginPage(req, res)
    } else if (method === 'GET' && pathname === '/page/register') {
      await registerPage(req, res)
    } else if (method === 'GET' && pathname === '/page/categories') {
      await categoriesPage(req, res)
    // } else if (method === 'GET' && pathname === '/page/subcategories') {
    
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: 'Route not found' }))
    }
}