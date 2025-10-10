import http from 'http'
import { URL } from 'url'
import { IncomingMessage, ServerResponse } from 'http'
import { getAll } from './getAll'
import { createCategory }  from './create'
import { getRequestBody } from '$/utils/getRequestBody'
import { CategoryInput } from '$/types/index'
import { getSession } from '$/utils/auth'


export async function handleCategory(
  req: IncomingMessage,
  res: ServerResponse,
  method: string | undefined,
  pathname: string) {

    if (method === 'GET' && pathname === '/api/category/getAll') {
      await handleGetALL(req, res)
    } else if (method === 'POST' && pathname === '/api/category/create') {
      await handleCreateCategory(req, res)
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: 'Route not found' }))
    }
}


async function handleGetALL(req: http.IncomingMessage, res: http.ServerResponse) {
  try {
    const result = await getAll()

    if (result.success) {
      res.writeHead(200, { 'Content-Type': 'application/json' })
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' })
    }
    res.end(JSON.stringify(result))
  } catch (error) {
    console.error('Categories handler error:', error)
    res.writeHead(500, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'Internal server error' }))
  }
}
async function handleCreateCategory(req: http.IncomingMessage, res: http.ServerResponse) {
  try {
    const body = await getRequestBody(req)
    const categoryData: CategoryInput = JSON.parse(body)
    const user = await getSession(req)
    console.log(user)

    // Валидация
    if (!categoryData.name) {
      res.writeHead(400, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: 'Имя обязательно для заполнения' }))
      return
    }

    const result = await createCategory(categoryData, user)

    if (result.success) {
      res.writeHead(200, { 'Content-Type': 'application/json' })
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' })
    }
    res.end(JSON.stringify(result))
  } catch (error) {
    console.error('Categories handler error:', error)
    res.writeHead(500, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'Internal server error' }))
  }
}