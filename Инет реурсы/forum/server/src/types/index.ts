import { Category } from '@prisma/client';

export interface UserInput {
  email: string
  username: string
  password: string
}

export interface LoginInput {
  email: string
  password: string
}

export interface AuthResponse {
  success: boolean
  message: string
  user?: {
    id: string
    email: string
    username: string
    role: string
  }
  token?: string
}

export interface GetAllCategoriesResponse {
  success: boolean;
  message: string;
  categories?: {
    data: Category[];
    count: number;
  };
}

export interface CategoryInput
{
  name: string
  description: string
  order: number
}

export interface CategoryResponse {
  success: boolean;
  message: string;
  category?: Category;
}
