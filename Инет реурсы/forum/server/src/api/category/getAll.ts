import { prisma } from '$/utils/database'; 
import { GetAllCategoriesResponse } from '$/types/index';

export async function getAll(): Promise<GetAllCategoriesResponse> {
  try {
    const categories = await prisma.category.findMany({
      orderBy: {
        name: 'asc',
      },
    });

    if (categories.length === 0) {
      return {
        success: false,
        message: 'Категории не найдены',
      };
    }

    return {
      success: true,
      message: 'Категории успешно получены',
      categories: {
        data: categories,
        count: categories.length,
      },
    };
  } catch (error) {
    console.error('Get categories error:', error);
    return {
      success: false,
      message: 'Ошибка при получении категорий',
    };
  }
}