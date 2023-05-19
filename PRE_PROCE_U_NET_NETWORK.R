
# PRE-PROCESAMIENTO IMAGENES 204031 PARA REDES NEURONALES U-NET
HUELLA <- "HUELLA_204030"
i <- 1

huella_list <- c("HUELLA_204030","HUELLA_204031","HUELLA_205030")

for (HUELLA in huella_list) {
  
  setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/Z_DATASET_DEFINITIVO_PERI/PERIMETROS_DEFINITIVO_",HUELLA))
  
  shape <- dir()
  shape <- str_subset(shape, "shp$")
  
  
  for (i in 1:length(shape)){
    
    j <- paste0(substr(shape[i], start = 1, stop = 8),".TIF")
    
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/DATOS_TOTAL/MULTI_SPECTRAL"))
    IMAG1 <- stack(j)
    IMAG1[is.na(IMAG1)] <- -9999
    
    # --------------------- UPLOAD PUNTOS DE INCENDIO ---------------------------- #
    k <- shape[i]
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/Z_DATASET_DEFINITIVO_PERI/PERIMETROS_DEFINITIVO_",HUELLA))
    POLIG <- st_read(k)
    
    # ---------------------------------------------------------------------------- #
    
    # --- GENERACION DE LA MASK-ARA --- #
    IMAG1_REC <- IMAG1[[3]]
    
    r <- rasterize(POLIG, IMAG1_REC)
    r[r > 0] <- 1
    
    r <- mask(r, IMAG1_REC)
    
    r[r > 0 ] <- 1
    r[is.na(r)] <- 0
    
    
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/MASK"))
    writeRaster(r,
                filename = j, 
                format = "GTiff",
                datatype= 'FLT4S')
    
    
    # =========================================================================================================================== #
    # --------------------------------------------------------------------------------------------------------------------------- #  
    # =========================================================================================================================== #  
    # ------------------------------------------------------- OBTENCION DE PATCH ------------------------------------------------ #
    # =========================================================================================================================== #  
    
    
    # --------------------- DETERMINACION DE CENTROIDES -------------------------- #
    centroids <- gCentroid(spgeom = methods::as( object = POLIG, Class = "Spatial" ), byid = TRUE)
    centroids<-st_as_sf(centroids)
    
    
    # ------------------ CREACION DE LA CARPETA CENTROIDES ----------------------- #
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/CENTROIDES"))
    dir.create(substr(j, start = 1, stop = 8))
    dir_centro <- paste0(getwd(),"/",substr(j, start = 1, stop = 8))
    
    
    # ----------------- CREACION DE LA CARPETA NUEVA HUELLAS --------------------- #
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/HUELLAS"))
    dir.create(substr(j, start = 1, stop = 8))
    dir_huella <- paste0(getwd(),"/",substr(j, start = 1, stop = 8))
    
    
    # -------------- CREACION DE LAS HUELLAS PARA CADA AREA QUEMADA -------------- #
    for (n in 1:nrow(centroids)){
      
      centro_n <- centroids$geometry[n]
      
      setwd(dir_centro)
      st_write(centro_n,
               paste0(substr(j, start = 1, stop = 8),"_",n,".shp")) 
      
      
      huella <- st_buffer(centro_n, dist=3840, endCapStyle = "SQUARE")
      
      setwd(dir_huella)
      st_write(huella,
               paste0(substr(j, start = 1, stop = 8),"_",n,".shp")) 
    }
    
    # ============================================================================ #
    # -------------- RECORTE DE CLOUD MASK E IMAG MULTIESPECTRAL ----------------- #  
    # ============================================================================ #
    
    # --------------- CREACION DE LA CARPETA NUEVA CLOUDS TRIM ------------------- #
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/CLOUDS"))
    dir.create(substr(j, start = 1, stop = 8))
    dir_nubes <- paste0(getwd(),"/",substr(j, start = 1, stop = 8))
    
    
    # -------------- CREACION DE LA CARPETA IMAGENES MULTIESPEC ------------------ #
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/RECORTE_IMAG"))
    dir.create(substr(j, start = 1, stop = 8))
    dir_imagRECORT <- paste0(getwd(),"/",substr(j, start = 1, stop = 8))
    
    # -------------------- UPLOAD NUBES Y MULTIESPECTRALES ----------------------- #
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/DATOS_TOTAL/REFINED_CLOUDS"))
    clouds <- raster(j)
    
    
    IMAGENES_MULTISPEC <- IMAG1
    
    
    # ------------------------ RECORTE DE LOS RASTER ----------------------------- #
    setwd(dir_huella)
    LISTA <- dir()
    LISTA <- str_subset(LISTA, "shp$")
    
    
    for (m in 1:length(LISTA)) {
      
      C <- LISTA[m]
      
      setwd(dir_huella)
      MASK_HUELLA <- st_read(C)
      
      clouds_MASK <- crop(clouds,MASK_HUELLA)
      IMAGENES_MULTISPEC_MASK <- crop(IMAGENES_MULTISPEC,MASK_HUELLA)
      
      setwd(dir_nubes)
      writeRaster(clouds_MASK,
                  filename = paste0(str_sub(C,1,nchar(C)-4),".TIF"), 
                  format = "GTiff",
                  datatype= 'FLT4S') 
      
      setwd(dir_imagRECORT)
      writeRaster(IMAGENES_MULTISPEC_MASK,
                  filename = paste0(str_sub(C,1,nchar(C)-4),".TIF"), 
                  format = "GTiff",
                  datatype= 'FLT4S') 
      
    }
    
    
    #============================================================================= #
    # ----------- OBTENCION DE MASCARAS DE INCENDIO EN LA HUELLA ----------------- #
    #============================================================================= #
    
    
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/MASK_PATCH"))
    dir.create(substr(j, start = 1, stop = 8))
    dir_mask_patch <- paste0(getwd(),"/",substr(j, start = 1, stop = 8))
    
    
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/MASCARAS_PATCH_CONJUNTO"))
    dir.create(substr(j, start = 1, stop = 8))
    dir_mask_patch_conxunto <- paste0(getwd(),"/",substr(j, start = 1, stop = 8))
    
    # ··· MASK POR INCENDIO SEPARADO Y EN CONJUNTO ··· #
    for (w in 1:length(LISTA)) {
      
      # ··· MASK POR INCENDIO SEPARADO ··· #
      LISTA[w]
      # --- INTERSECCION DE CENTROIDE CON SU RESPECTIVO POLIGONO --- #
      setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/CENTROIDES/",substr(j, start = 1, stop = 8)))
      CENTRO_OVER <- st_read(LISTA[w])
      
      SUPER_POSIC <- POLIG[CENTRO_OVER,]  
      
      # --- CARGAR IMAGEN RECORTADA CON HUELLA PARA DETERMINAR EXTENSION DE MASCARA --- #
      setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/RECORTE_IMAG/",substr(LISTA[w], start = 1, stop = 8),"/"))
      IMAG_to_TRIM <- raster(paste0(str_sub(LISTA[w],1,nchar(LISTA[w])-4),".TIF"))
      
      
      if (nrow(SUPER_POSIC) == 0) {
        
        CENTRO_OVER.buffer<-st_buffer(CENTRO_OVER,500) 
        SUPER_POSIC <-POLIG[CENTRO_OVER.buffer,]
        
      }
      
      
      # --- RASTERIZACION DE LOS POLIGONOS --- #
      r <- rasterize(SUPER_POSIC, IMAG_to_TRIM)
      r[r > 0 ] <- 255
      r[is.na(r)] <- 0
      
      
      setwd(dir_mask_patch)
      writeRaster(r,
                  filename = paste0(str_sub(LISTA[w],1,nchar(LISTA[w])-4),".TIF"), 
                  format = "GTiff",
                  datatype= 'FLT4S')
      
      # -----------------------------------#
      
      # ··· MASK POR INCENDIO CONJUNTO ··· #
      
      r <- rasterize(POLIG, IMAG_to_TRIM)
      r[r > 0 ] <- 255
      r[is.na(r)] <- 0
      
      
      setwd(dir_mask_patch_conxunto)
      writeRaster(r,
                  filename = paste0(str_sub(LISTA[w],1,nchar(LISTA[w])-4),".TIF"), 
                  format = "GTiff",
                  datatype= 'FLT4S')
      
      
      
      # NOTA: REVISAR SI APLICAR MASK POR INCENDIO SEPARADO O CON EL CONJUNTO PRESENTE #
      
    }   
    
    print("NEXT")
  }
  
}


#_______________________________________________________________________________ #
# ============================================================================== #
# ------------------------------------------------------------------------------ #
# ============================================================================== #
# -------- RECORTES DE IMAGENES DEFINITIVAS CON BANDAS EXTRA Y ENMASK ---------- #
# ============================================================================== #
# ------------------------------------------------------------------------------ #

# -------- OBTENCION DE LAS FECHAS (FOLDER) DE LAS IMAGENES ORIGINALES --------- #
for (HUELLA in huella_list) {
  
  setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/RECORTE_IMAG"))
  LISTA2 <- dir()

  for (x in LISTA2) {
  
    # --- CREACION DE LAS CARPETAS --- #
  
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/1_RESULTADOS/MULTISPECTRAL_TOTAL"))
    dir.create(x)
    dir_multispect_TOTAL<- paste0(getwd(),"/",x)
  
  
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/1_RESULTADOS/MULTISPECTRAL_ENMASK"))
    dir.create(x)
    dir_multispect_mask<- paste0(getwd(),"/",x)
  
  
    # --- LISTA DE LOS ARCHIVOS N POR CADA AÑO DE IMAGEN --- #
  
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/RECORTE_IMAG/",x))
    LIST_IN_X <- dir()
  
    for (y in LIST_IN_X){
      setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/RECORTE_IMAG/",x))
    
      # --- OBTAINING OF NEW LAYERS FROM NDVI AND NBR --- #
      IMAX <- stack(y)
    
      NIR <- IMAX[[3]] 
      RED <- IMAX[[4]] 
      SWIR2 <- IMAX[[1]] 
    
      NDVI <- (NIR-RED)/(NIR+RED)
      NDVI[NDVI > 1] <- 1
      NDVI[NDVI < -1] <- -1
    
      NBR <- (NIR-SWIR2)/(NIR+SWIR2)
      NBR[NBR > 1] <- 1
      NBR[NBR < -1] <- -1
    
      IMAG_COMPLETADA <- stack(IMAX, NDVI, NBR)
    
      # ---------- #
      if (nrow(IMAG_COMPLETADA) < 256 || ncol(IMAG_COMPLETADA) < 256) {
      
        xmaxima <- xmax(IMAG_COMPLETADA)
        xminima <- xmin(IMAG_COMPLETADA)
        ymaxima <- ymax(IMAG_COMPLETADA)
        yminima <- ymin(IMAG_COMPLETADA)
      
        x_pto <- xminima 
        y_pto <- ymaxima 
      
        xmaxima <- x_pto + (256*30)
        xminima <- x_pto 
        ymaxima <- y_pto 
        yminima <- y_pto - (256*30)
      
      
        base_raster <- raster(ncol= 256, nrow= 256, xmx= xmaxima, xmn= xminima, ymx= ymaxima, ymn= yminima)
      
        proj4string(base_raster) <- 
          CRS("+proj=utm +zone=29 +north +ellps=WGS84 +datum=WGS84")
      
        values(base_raster) <- as.integer(runif(ncell(base_raster),
                                                min = -9999,
                                                max = -9999))
      
        SWIR2 <- IMAG_COMPLETADA[[1]]
        SWIR1 <- IMAG_COMPLETADA[[2]]
        NIR <- IMAG_COMPLETADA[[3]]
        RED <- IMAG_COMPLETADA[[4]]
        GREEN <- IMAG_COMPLETADA[[5]]
        BLUE <- IMAG_COMPLETADA[[6]]
        NDVI <- IMAG_COMPLETADA[[7]]
        NBR <- IMAG_COMPLETADA[[8]]
      
        SWIR2 <- raster::merge(SWIR2, base_raster, tolerance = 0.05)
        SWIR1 <- raster::merge(SWIR1, base_raster, tolerance = 0.05)
        NIR <- raster::merge(NIR, base_raster, tolerance = 0.05)
        RED <- raster::merge(RED, base_raster, tolerance = 0.05)
        GREEN <- raster::merge(GREEN, base_raster, tolerance = 0.05)
        BLUE <- raster::merge(BLUE, base_raster, tolerance = 0.05)
        NDVI <- raster::merge(NDVI, base_raster, tolerance = 0.05)
        NBR <- raster::merge(NBR, base_raster, tolerance = 0.05)
      
      
        IMAG_COMPLETADA <- stack(SWIR2, SWIR1, NIR, RED, GREEN, BLUE, NDVI, NBR)
      
      }
      # ---------- #
      # --- TRANSFORMACION A ESCALA DE 0-255 --- #
      
      a0 <- IMAG_COMPLETADA[[1]]
      a0[a0 < 0]<-NA 
      a0[a0 > 1]<-NA 
      a0 <- a0*255
      
      b0 <- IMAG_COMPLETADA[[2]]
      b0[b0 < 0]<-NA 
      b0[b0 > 1]<-NA
      b0 <- b0*255
      
      c0 <- IMAG_COMPLETADA[[3]]
      c0[c0 < 0]<-NA 
      c0[c0 > 1]<-NA
      c0 <- c0*255
      
      d0 <- IMAG_COMPLETADA[[4]]
      d0[d0 < 0]<-NA 
      d0[d0 > 1]<-NA
      d0 <- d0*255
      
      e0 <- IMAG_COMPLETADA[[5]]
      e0[e0 < 0]<-NA 
      e0[e0 > 1]<-NA
      e0 <- e0*255
      
      f0 <- IMAG_COMPLETADA[[6]]
      f0[f0 < 0]<-NA 
      f0[f0 > 1]<-NA
      f0 <- f0*255
      
      g0 <- IMAG_COMPLETADA[[7]]
      g0[g0 < -1]<-NA 
      g0[g0 > 1]<-NA 
      g0[g0 >= 0 & g0 <= 1]<- g0[g0 >= 0 & g0 <= 1]+1
      g0[g0 < 0 & g0 >= -1]<- g0[g0 < 0 & g0 >= -1]+1 
      g0[g0 < 0]<-NA 
      g0[g0 > 2]<-NA 
      g0 <- g0*127.5
      
      h0 <- IMAG_COMPLETADA[[8]]
      h0[h0 < -1]<- NA
      h0[h0 > 1]<-NA
      h0[h0 >= 0 & h0 <= 1]<- h0[h0 >= 0 & h0 <= 1]+1
      h0[h0 < 0 & h0 >= -1]<- h0[h0 < 0 & h0 >= -1]+1 
      h0[h0 < 0]<-NA 
      h0[h0 > 2]<-NA 
      h0 <- h0*127.5
      
      IMAG_COMPLETADA <- stack(a0,b0,c0,d0,e0,f0,g0,h0)
      
      labels <- c("SWIR2","SWIR1", "NIR", "RED", "GREEN", "BLUE", "NDVI", "NBR")
      names(IMAG_COMPLETADA) <- labels
    
      setwd(dir_multispect_TOTAL)
    
      writeRaster(IMAG_COMPLETADA,
                  filename = y,
                  format = "GTiff",
                  datatype= 'FLT4S')
    
    
    
      # --- TRIM OF MULTISPECTRAL IMAGES --- #
    
      setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/CLOUDS/", x))
      MASKARA <- raster(y)
    
      # ---------- #
      if (nrow(MASKARA) < 256 || ncol(MASKARA) < 256) {
      
        xmaxima2 <- xmax(MASKARA)
        xminima2 <- xmin(MASKARA)
        ymaxima2 <- ymax(MASKARA)
        yminima2 <- ymin(MASKARA)
      
        x_pto2 <- xminima2 
        y_pto2 <- ymaxima2 
      
        xmaxima2 <- x_pto2 + (256*30)
        xminima2 <- x_pto2 
        ymaxima2 <- y_pto2 
        yminima2 <- y_pto2 - (256*30)
      
      
        base_raster2 <- raster(ncol= 256, nrow= 256, xmx= xmaxima2, xmn= xminima2, ymx= ymaxima2, ymn= yminima2)
      
        proj4string(base_raster2) <- 
            CRS("+proj=utm +zone=29 +north +ellps=WGS84 +datum=WGS84")
      
        values(base_raster2) <- as.integer(runif(ncell(base_raster2),
                                                min = -9999,
                                                max = -9999))
      
        MASKARA <- raster::merge(MASKARA, base_raster2, tolerance = 0.05)
      }
    
      # ---------- #
    
      MASKARA[MASKARA < 0.5] <- NA 
  
      IMAG_COMPLETADA <- mask(IMAG_COMPLETADA,MASKARA)
  
      setwd(dir_multispect_mask)
      writeRaster(IMAG_COMPLETADA,
                  filename = y,
                  format = "GTiff",
                  datatype= 'FLT4S')
  

    }
  }
}




# ====================================== #
# --- RESIZE TOTAL SET OF MASK PATCH --- #
# ====================================== #


for (HUELLA in huella_list) {
  
  setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/MASCARAS_PATCH_CONJUNTO"))
  LISTA2 <- dir()
  
  for (x in LISTA2) {
    
    # --- CREACION DE LAS CARPETAS --- #
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/1_RESULTADOS/MASK_FIRE"))
    dir.create(x)
    dir_MASK_FIRE<- paste0(getwd(),"/",x)
    
    # --- LISTA DE LOS ARCHIVOS N POR CADA AÑO DE IMAGEN --- #
    setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/MASCARAS_PATCH_CONJUNTO/",x))
    LIST_IN_X <- dir()
    
    for (y in LIST_IN_X){
      
      setwd(paste0("C:/PFIRE_FOREST_GALI_U_NET/",HUELLA,"/MASCARAS_PATCH_CONJUNTO/", x))
      MASKARA <- raster(y)
      
      # ---------- #
      if (nrow(MASKARA) < 256 || ncol(MASKARA) < 256) {
        
        xmaxima2 <- xmax(MASKARA)
        xminima2 <- xmin(MASKARA)
        ymaxima2 <- ymax(MASKARA)
        yminima2 <- ymin(MASKARA)
        
        x_pto2 <- xminima2 
        y_pto2 <- ymaxima2 
        
        xmaxima2 <- x_pto2 + (256*30)
        xminima2 <- x_pto2 
        ymaxima2 <- y_pto2 
        yminima2 <- y_pto2 - (256*30)
        
        
        base_raster2 <- raster(ncol= 256, nrow= 256, xmx= xmaxima2, xmn= xminima2, ymx= ymaxima2, ymn= yminima2)
        
        proj4string(base_raster2) <- 
          CRS("+proj=utm +zone=29 +north +ellps=WGS84 +datum=WGS84")
        
        values(base_raster2) <- as.integer(runif(ncell(base_raster2),
                                                 min = 0,
                                                 max = 0))
        
        MASKARA <- raster::merge(MASKARA, base_raster2, tolerance = 0.05)
      }
      
      # ---------- #
      
      MASKARA[MASKARA < 0] <- NA 
      MASKARA[MASKARA > 255] <- NA 
      
      setwd(dir_MASK_FIRE)
      writeRaster(MASKARA,
                  filename = y,
                  format = "GTiff",
                  datatype= 'FLT4S')
      
      
      
    }
  }
}





