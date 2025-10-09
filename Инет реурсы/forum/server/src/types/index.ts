import { Category, Subcategory } from '@prisma/client';

export interface UserInput {
  email: string
  username: string
  password: string
}

export interface AuthUser {
  id: string;
  email: string;
  username: string;
  role: UserRole;
}

export type UserRole = 'USER' | 'MODERATOR' | 'ADMIN'

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

export interface CategoriesResponse {
  success: boolean;
  message: string;
  categories?: {
    data: Category[];
    count: number;
  };
}

export interface SubcategoriesResponse {
  success: boolean;
  message: string;
  subcategories?: {
    data: Subcategory[];
    count: number;
  };
}

export interface CategoryInput
{
  name: string
  description?: string
  order?: number
}

export interface SubcategoryInput
{
  name: string
  categoryId: string
  description?: string
  order?: number
}

export interface CategoryResponse {
  success: boolean;
  message: string;
  category?: Category;
}

export interface SubcategoryResponse {
  success: boolean;
  message: string;
  subcategory?: Subcategory;
}

