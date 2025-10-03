import { prisma } from '../utils/database'
import { LoginInput, AuthResponse } from '../types/index'
import { verifyPassword, generateToken } from './utils'

export async function login(loginData: LoginInput): Promise<AuthResponse> {
  try {
    const { email, password } = loginData

    // Поиск пользователя
    const user = await prisma.user.findUnique({
      where: { email }
    })

    if (!user) {
      return {
        success: false,
        message: 'Неверный email или пароль'
      }
    }

    // Проверка пароля
    const isPasswordValid = await verifyPassword(password, user.password)

    if (!isPasswordValid) {
      return {
        success: false,
        message: 'Неверный email или пароль'
      }
    }

    // Генерация токена
    const token = generateToken()

    // Создание сессии
    await prisma.session.create({
      data: {
        userId: user.id,
        token,
        expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000) // 7 дней
      }
    })

    return {
      success: true,
      message: 'Успешный вход',
      user: {
        id: user.id,
        email: user.email,
        username: user.username,
        role: user.role
      },
      token
    }

  } catch (error) {
    console.error('Login error:', error)
    return {
      success: false,
      message: 'Ошибка при входе'
    }
  }
}