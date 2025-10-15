import { Grid, Box, Typography } from '@mui/material';
import ProductCard from './ProductCard';

export default function ResultsGrid({ results }) {
    return (
        <Box sx={{ mt: 4 }}>
            <Typography variant="h5" component="h3" gutterBottom sx={{ fontWeight: 'bold' }}>
                Results:
            </Typography>
            <Grid container spacing={3}>
                {results.map((product) => (
                    <Grid item key={product.product_id} xs={6} sm={4} md={3}>
                        <ProductCard product={product} />
                    </Grid>
                ))}
            </Grid>
        </Box>
    );
}