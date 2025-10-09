import { AuthUser } from '$/types/index';

export function requireAuth(user?: AuthUser) {
  if (!user) {
    throw new Error('Необходима авторизация');
  }
  return user;
}

export function requireAdmin(user?: AuthUser) {
  const authUser = requireAuth(user);
  
  if (authUser.role !== 'ADMIN') {
    throw new Error('Недостаточно прав. Требуется роль ADMIN');
  }
  
  return authUser;
}