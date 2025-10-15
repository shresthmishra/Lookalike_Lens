import { Card, CardMedia, CardContent, Typography } from '@mui/material';

export default function ProductCard({ product }) {
    const similarity = (1 / (1 + product.similarity_score)) * 100;

    return (
        <Card sx={{ height: '100%' }}>
            <CardMedia
                component="img"
                alt={product.name}
                height="200"
                image={product.image_url}
                sx={{ objectFit: 'cover' }}
            />
            <CardContent>
                <Typography gutterBottom variant="body2" color="text.secondary">
                    {product.category}
                </Typography>
                <Typography variant="h6" component="div" noWrap title={product.name}>
                    {product.name}
                </Typography>
                <Typography variant="body1" color="green" sx={{ fontWeight: 'bold', mt: 1 }}>
                    {similarity.toFixed(1)}% Match
                </Typography>
            </CardContent>
        </Card>
    );
}