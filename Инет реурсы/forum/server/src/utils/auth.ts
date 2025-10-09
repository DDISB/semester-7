import { IncomingMessage } from 'http';
import { prisma } from './database';
import { AuthUser } from '$/types/index';

export async function getSession(req: IncomingMessage): Promise<AuthUser | undefined> {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');
    console.log('token - ', token)
    
    if (!token) {
      return undefined;
    }

    // Проверяем сессию в БД
    const session = await prisma.session.findFirst({
      where: {
        token,
        expiresAt: {
          gt: new Date() // Сессия не просрочена
        }
      },
      include: {
        user: {
          select: {
            id: true,
            email: true,
            username: true,
            role: true,
            avatar: true
          }
        }
      }
    });

    if (!session) {
      return undefined;
    }

    // Обновляем время истечения сессии (опционально)
    await prisma.session.update({
      where: { id: session.id },
      data: { 
        expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000) 
      }
    });

    return session.user;

  } catch (error) {
    console.error('Get session error:', error);
    return undefined;
  }
}