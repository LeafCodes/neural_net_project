def running_average(new_value, current_average, n):
    """
    Menghitung rata-rata yang diperbarui setiap kali ada data baru.
    
    Args:
    new_value (float): Nilai baru yang ditambahkan ke dalam kumpulan data.
    current_average (float): Rata-rata sebelumnya dari semua data.
    n (int): Jumlah data sebelum nilai baru ditambahkan.
    
    Returns:
    float: Rata-rata yang diperbarui setelah nilai baru ditambahkan.
    """
    updated_average = (current_average * n + new_value) / (n + 1)
    return updated_average