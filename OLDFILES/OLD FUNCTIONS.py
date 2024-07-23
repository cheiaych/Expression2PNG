#OLD FUNCTIONS
def scaling_ratio(img, max_width, max_height):
     #img.shape[1] #width
     #img.shape[0] #length

     ratio = min(max_width / img.shape[1], max_height / img.shape[0])
     #print(img.shape[1]*ratio)
     return ratio

def flatten_img(mat):
    return numpy.true_divide(numpy.asfarray(numpy.ravel(numpy.flip(mat,2)), dtype='f'), 255.0)

def setup_dpg_window():

    load_expressions()

    dpg.create_context()

    """
    width, height, channels, data = dpg.load_image('./TestImages/neutral.jpg')
    temp_img = image(data, width, height)

    print(data == temp_img.data)

    with dpg.texture_registry(show=True):
        expression_texture_id = dpg.add_dynamic_texture(width = temp_img.width, height = temp_img.height, default_value = temp_img.data)
    

    with dpg.texture_registry(show = False):
        expression_texture_id = dpg.add_dynamic_texture(width = expressions['neutral'].width, height = expressions['neutral'].height, default_value = expressions['neutral'].data)

    with dpg.window(tag = 'expression_display', autosize = True):
        dpg.add_image(expression_texture_id, tag = 'expression')
    """
    with dpg.window(tag = 'expression_display', autosize = True):
        dpg.add_text('Placeholder')

    with dpg.window(tag = 'control_display', no_close = True, autosize = True):
            
            dpg.add_button(label="Test", callback=test_display)

            dpg.add_text('Sensitivities')
            dpg.add_slider_float(label = 'Angry', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(label = 'Disgust', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(label = 'Fear', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(label = 'Happy', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(label = 'Sad', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(label = 'Suprise', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(label = 'Neutral', default_value = 0.0, max_value = 100.0, min_value = -100.0)

    with dpg.item_handler_registry(tag = '#resize_handler'):
        dpg.add_item_resize_handler(callback = window_resized())

    dpg.set_primary_window('expression_display', True)

    dpg.create_viewport(title='Window 1', width=1024, height=576)

    display_expression('neutral')

#OLD FUNCTIONS
def custom_analysis(db_path,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,
    time_threshold=5,
    frame_threshold=5):

    print("Test")
    cap = cv2.VideoCapture(source)  # webcam
    while True:
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        _, cap_img = cap.read()
        #cv2.imshow("Output", cap_img)
        faces = DeepFace.extract_faces(
            img_path=cap_img,
            target_size=functions.find_target_size(model_name=model_name),
            detector_backend=detector_backend,
            enforce_detection=False
        )
        face_dimensions = faces[0]["facial_area"]
        #print(face_dimensions)

        cropped_face = cap_img[int(face_dimensions["y"]):int(face_dimensions["y"] + face_dimensions["h"]), int(face_dimensions["x"]):int(face_dimensions["x"] + face_dimensions["w"])]
        cv2.imshow("Output", cropped_face)
        
        demographies = DeepFace.analyze(
                                img_path=cropped_face,
                                actions=("emotion"),
                                detector_backend=detector_backend,
                                enforce_detection=False,
                                silent=True,
                            )
        
        sensitivities = {
            "angry": 0.1,
            "disgust": 0.1,
            "fear": 0.1,
            "happy": 1.0,
            "sad": 0.1,
            "surprise": 1.0,
            "neutral": 1.0,
        }
        for key in list(sensitivities.keys()):
            demographies[0]["emotion"][key] = demographies[0]["emotion"][key] * sensitivities[key]
        
        emotion = max(demographies[0]["emotion"], key=demographies[0]["emotion"].get)
        #emotion = demographies[0]["emotion"]
        print(emotion)

        #img = cv2.imread("./Testimages/" + emotion + ".jpg")
        #img = cv2.resize(img, (500,500))
        #cv2.imshow("Output", img)

        if cv2.waitKey(500) & 0xFF == ord('q'):  #press q to quit, waitKey sets refresh rate in ms (waitKey(1000) is 1 frame a second), 
            break

    cap.release()
    cv2.destroyAllWindows()
   
def analysis(
    db_path,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,
    time_threshold=5,
    frame_threshold=5,
):
    # global variables
    text_color = (255, 255, 255)
    pivot_img_size = 112  # face recognition result image

    enable_emotion = True
    enable_age_gender = True
    # ------------------------
    # find custom values for this input set
    target_size = functions.find_target_size(model_name=model_name)
    # ------------------------
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    DeepFace.build_model(model_name=model_name)
    logger.info(f"facial recognition model {model_name} is just built")

    if enable_face_analysis:
        DeepFace.build_model(model_name="Age")
        logger.info("Age model is just built")
        DeepFace.build_model(model_name="Gender")
        logger.info("Gender model is just built")
        DeepFace.build_model(model_name="Emotion")
        logger.info("Emotion model is just built")
    # -----------------------
    # call a dummy find function for db_path once to create embeddings in the initialization
    DeepFace.find(
        img_path=np.zeros([224, 224, 3]),
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=False,
    )
    # -----------------------
    # visualization
    freeze = False
    face_detected = False
    face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
    freezed_frame = 0
    tic = time.time()

    cap = cv2.VideoCapture(source)  # webcam
    while True:
        _, img = cap.read()

        if img is None:
            break

        # cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        raw_img = img.copy()
        resolution_x = img.shape[1]
        resolution_y = img.shape[0]

        if freeze == False:
            try:
                # just extract the regions to highlight in webcam
                face_objs = DeepFace.extract_faces(
                    img_path=img,
                    target_size=target_size,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                )
                faces = []
                for face_obj in face_objs:
                    facial_area = face_obj["facial_area"]
                    faces.append(
                        (
                            facial_area["x"],
                            facial_area["y"],
                            facial_area["w"],
                            facial_area["h"],
                        )
                    )
            except:  # to avoid exception if no face detected
                faces = []

            if len(faces) == 0:
                face_included_frames = 0
        else:
            faces = []

        detected_faces = []
        face_index = 0
        for x, y, w, h in faces:
            if w > 130:  # discard small detected faces

                face_detected = True
                if face_index == 0:
                    face_included_frames = (
                        face_included_frames + 1
                    )  # increase frame for a single face

                cv2.rectangle(
                    img, (x, y), (x + w, y + h), (67, 67, 67), 1
                )  # draw rectangle to main image

                cv2.putText(
                    img,
                    str(frame_threshold - face_included_frames),
                    (int(x + w / 4), int(y + h / 1.5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (255, 255, 255),
                    2,
                )

                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]  # crop detected face

                # -------------------------------------

                detected_faces.append((x, y, w, h))
                face_index = face_index + 1

                # -------------------------------------

        if face_detected == True and face_included_frames == frame_threshold and freeze == False:
            freeze = True
            # base_img = img.copy()
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()

        if freeze == True:

            toc = time.time()
            if (toc - tic) < time_threshold:

                if freezed_frame == 0:
                    freeze_img = base_img.copy()
                    # here, np.uint8 handles showing white area issue
                    # freeze_img = np.zeros(resolution, np.uint8)

                    for detected_face in detected_faces_final:
                        x = detected_face[0]
                        y = detected_face[1]
                        w = detected_face[2]
                        h = detected_face[3]

                        cv2.rectangle(
                            freeze_img, (x, y), (x + w, y + h), (67, 67, 67), 1
                        )  # draw rectangle to main image

                        # -------------------------------
                        # extract detected face
                        custom_face = base_img[y : y + h, x : x + w]
                        # -------------------------------
                        # facial attribute analysis

                        if enable_face_analysis == True:

                            demographies = DeepFace.analyze(
                                img_path=custom_face,
                                detector_backend=detector_backend,
                                enforce_detection=False,
                                silent=True,
                            )

                            if len(demographies) > 0:
                                # directly access 1st face cos img is extracted already
                                demography = demographies[0]

                                if enable_emotion:
                                    emotion = demography["emotion"]
                                    emotion_df = pd.DataFrame(
                                        emotion.items(), columns=["emotion", "score"]
                                    )
                                    emotion_df = emotion_df.sort_values(
                                        by=["score"], ascending=False
                                    ).reset_index(drop=True)

                                    # background of mood box

                                    # transparency
                                    overlay = freeze_img.copy()
                                    opacity = 0.4

                                    if x + w + pivot_img_size < resolution_x:
                                        # right
                                        cv2.rectangle(
                                            freeze_img
                                            # , (x+w,y+20)
                                            ,
                                            (x + w, y),
                                            (x + w + pivot_img_size, y + h),
                                            (64, 64, 64),
                                            cv2.FILLED,
                                        )

                                        cv2.addWeighted(
                                            overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img
                                        )

                                    elif x - pivot_img_size > 0:
                                        # left
                                        cv2.rectangle(
                                            freeze_img
                                            # , (x-pivot_img_size,y+20)
                                            ,
                                            (x - pivot_img_size, y),
                                            (x, y + h),
                                            (64, 64, 64),
                                            cv2.FILLED,
                                        )

                                        cv2.addWeighted(
                                            overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img
                                        )

                                    for index, instance in emotion_df.iterrows():
                                        current_emotion = instance["emotion"]
                                        emotion_label = f"{current_emotion} "
                                        emotion_score = instance["score"] / 100

                                        bar_x = 35  # this is the size if an emotion is 100%
                                        bar_x = int(bar_x * emotion_score)

                                        if x + w + pivot_img_size < resolution_x:

                                            text_location_y = y + 20 + (index + 1) * 20
                                            text_location_x = x + w

                                            if text_location_y < y + h:
                                                cv2.putText(
                                                    freeze_img,
                                                    emotion_label,
                                                    (text_location_x, text_location_y),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5,
                                                    (255, 255, 255),
                                                    1,
                                                )

                                                cv2.rectangle(
                                                    freeze_img,
                                                    (x + w + 70, y + 13 + (index + 1) * 20),
                                                    (
                                                        x + w + 70 + bar_x,
                                                        y + 13 + (index + 1) * 20 + 5,
                                                    ),
                                                    (255, 255, 255),
                                                    cv2.FILLED,
                                                )

                                        elif x - pivot_img_size > 0:

                                            text_location_y = y + 20 + (index + 1) * 20
                                            text_location_x = x - pivot_img_size

                                            if text_location_y <= y + h:
                                                cv2.putText(
                                                    freeze_img,
                                                    emotion_label,
                                                    (text_location_x, text_location_y),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5,
                                                    (255, 255, 255),
                                                    1,
                                                )

                                                cv2.rectangle(
                                                    freeze_img,
                                                    (
                                                        x - pivot_img_size + 70,
                                                        y + 13 + (index + 1) * 20,
                                                    ),
                                                    (
                                                        x - pivot_img_size + 70 + bar_x,
                                                        y + 13 + (index + 1) * 20 + 5,
                                                    ),
                                                    (255, 255, 255),
                                                    cv2.FILLED,
                                                )

                                if enable_age_gender:
                                    apparent_age = demography["age"]
                                    dominant_gender = demography["dominant_gender"]
                                    gender = "M" if dominant_gender == "Man" else "W"
                                    logger.debug(f"{apparent_age} years old {dominant_gender}")
                                    analysis_report = str(int(apparent_age)) + " " + gender

                                    # -------------------------------

                                    info_box_color = (46, 200, 255)

                                    # top
                                    if y - pivot_img_size + int(pivot_img_size / 5) > 0:

                                        triangle_coordinates = np.array(
                                            [
                                                (x + int(w / 2), y),
                                                (
                                                    x + int(w / 2) - int(w / 10),
                                                    y - int(pivot_img_size / 3),
                                                ),
                                                (
                                                    x + int(w / 2) + int(w / 10),
                                                    y - int(pivot_img_size / 3),
                                                ),
                                            ]
                                        )

                                        cv2.drawContours(
                                            freeze_img,
                                            [triangle_coordinates],
                                            0,
                                            info_box_color,
                                            -1,
                                        )

                                        cv2.rectangle(
                                            freeze_img,
                                            (
                                                x + int(w / 5),
                                                y - pivot_img_size + int(pivot_img_size / 5),
                                            ),
                                            (x + w - int(w / 5), y - int(pivot_img_size / 3)),
                                            info_box_color,
                                            cv2.FILLED,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            analysis_report,
                                            (x + int(w / 3.5), y - int(pivot_img_size / 2.1)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (0, 111, 255),
                                            2,
                                        )

                                    # bottom
                                    elif (
                                        y + h + pivot_img_size - int(pivot_img_size / 5)
                                        < resolution_y
                                    ):

                                        triangle_coordinates = np.array(
                                            [
                                                (x + int(w / 2), y + h),
                                                (
                                                    x + int(w / 2) - int(w / 10),
                                                    y + h + int(pivot_img_size / 3),
                                                ),
                                                (
                                                    x + int(w / 2) + int(w / 10),
                                                    y + h + int(pivot_img_size / 3),
                                                ),
                                            ]
                                        )

                                        cv2.drawContours(
                                            freeze_img,
                                            [triangle_coordinates],
                                            0,
                                            info_box_color,
                                            -1,
                                        )

                                        cv2.rectangle(
                                            freeze_img,
                                            (x + int(w / 5), y + h + int(pivot_img_size / 3)),
                                            (
                                                x + w - int(w / 5),
                                                y + h + pivot_img_size - int(pivot_img_size / 5),
                                            ),
                                            info_box_color,
                                            cv2.FILLED,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            analysis_report,
                                            (x + int(w / 3.5), y + h + int(pivot_img_size / 1.5)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (0, 111, 255),
                                            2,
                                        )

                        # --------------------------------
                        # face recognition
                        # call find function for custom_face

                        dfs = DeepFace.find(
                            img_path=custom_face,
                            db_path=db_path,
                            model_name=model_name,
                            detector_backend=detector_backend,
                            distance_metric=distance_metric,
                            enforce_detection=False,
                            silent=True,
                        )

                        if len(dfs) > 0:
                            # directly access 1st item because custom face is extracted already
                            df = dfs[0]

                            if df.shape[0] > 0:
                                candidate = df.iloc[0]
                                label = candidate["identity"]

                                # to use this source image as is
                                display_img = cv2.imread(label)
                                # to use extracted face
                                source_objs = DeepFace.extract_faces(
                                    img_path=label,
                                    target_size=(pivot_img_size, pivot_img_size),
                                    detector_backend=detector_backend,
                                    enforce_detection=False,
                                    align=False,
                                )

                                if len(source_objs) > 0:
                                    # extract 1st item directly
                                    source_obj = source_objs[0]
                                    display_img = source_obj["face"]
                                    display_img *= 255
                                    display_img = display_img[:, :, ::-1]
                                # --------------------
                                label = label.split("/")[-1]

                                try:
                                    if (
                                        y - pivot_img_size > 0
                                        and x + w + pivot_img_size < resolution_x
                                    ):
                                        # top right
                                        freeze_img[
                                            y - pivot_img_size : y,
                                            x + w : x + w + pivot_img_size,
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x + w, y),
                                            (x + w + pivot_img_size, y + 20),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x + w, y + 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y),
                                            (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                                            (x + w, y - int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )

                                    elif (
                                        y + h + pivot_img_size < resolution_y
                                        and x - pivot_img_size > 0
                                    ):
                                        # bottom left
                                        freeze_img[
                                            y + h : y + h + pivot_img_size,
                                            x - pivot_img_size : x,
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x - pivot_img_size, y + h - 20),
                                            (x, y + h),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x - pivot_img_size, y + h - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y + h),
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (x, y + h + int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )

                                    elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                                        # top left
                                        freeze_img[
                                            y - pivot_img_size : y, x - pivot_img_size : x
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x - pivot_img_size, y),
                                            (x, y + 20),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x - pivot_img_size, y + 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y),
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y - int(pivot_img_size / 2),
                                            ),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y - int(pivot_img_size / 2),
                                            ),
                                            (x, y - int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )

                                    elif (
                                        x + w + pivot_img_size < resolution_x
                                        and y + h + pivot_img_size < resolution_y
                                    ):
                                        # bottom righ
                                        freeze_img[
                                            y + h : y + h + pivot_img_size,
                                            x + w : x + w + pivot_img_size,
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x + w, y + h - 20),
                                            (x + w + pivot_img_size, y + h),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x + w, y + h - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y + h),
                                            (
                                                x + int(w / 2) + int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (
                                                x + int(w / 2) + int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (x + w, y + h + int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )
                                except Exception as err:  # pylint: disable=broad-except
                                    logger.error(str(err))

                        tic = time.time()  # in this way, freezed image can show 5 seconds

                        # -------------------------------

                time_left = int(time_threshold - (toc - tic) + 1)

                cv2.rectangle(freeze_img, (10, 10), (90, 50), (67, 67, 67), -10)
                cv2.putText(
                    freeze_img,
                    str(time_left),
                    (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow("img", freeze_img)

                freezed_frame = freezed_frame + 1
            else:
                face_detected = False
                face_included_frames = 0
                freeze = False
                freezed_frame = 0

        else:
            cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()

def analysis_emotion(
    db_path,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,
    time_threshold=5,
    frame_threshold=5,
):
    # global variables
    text_color = (255, 255, 255)
    pivot_img_size = 112  # face recognition result image

    enable_emotion = True
    enable_age_gender = False
    # ------------------------
    # find custom values for this input set
    target_size = functions.find_target_size(model_name=model_name)
    # ------------------------
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    DeepFace.build_model(model_name=model_name)
    logger.info(f"facial recognition model {model_name} is just built")

    if enable_face_analysis:
        DeepFace.build_model(model_name="Age")
        logger.info("Age model is just built")
        DeepFace.build_model(model_name="Gender")
        logger.info("Gender model is just built")
        DeepFace.build_model(model_name="Emotion")
        logger.info("Emotion model is just built")
    # -----------------------
    # call a dummy find function for db_path once to create embeddings in the initialization
    DeepFace.find(
        img_path=np.zeros([224, 224, 3]),
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=False,
    )
    # -----------------------
    # visualization
    freeze = False
    face_detected = False
    face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
    freezed_frame = 0
    tic = time.time()

    cap = cv2.VideoCapture(source)  # webcam
    while True:
        _, img = cap.read()

        if img is None:
            break

        # cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        raw_img = img.copy()
        resolution_x = img.shape[1]
        resolution_y = img.shape[0]

        if freeze == False:
            try:
                # just extract the regions to highlight in webcam
                face_objs = DeepFace.extract_faces(
                    img_path=img,
                    target_size=target_size,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                )
                faces = []
                for face_obj in face_objs:
                    facial_area = face_obj["facial_area"]
                    faces.append(
                        (
                            facial_area["x"],
                            facial_area["y"],
                            facial_area["w"],
                            facial_area["h"],
                        )
                    )
            except:  # to avoid exception if no face detected
                faces = []

            if len(faces) == 0:
                face_included_frames = 0
        else:
            faces = []

        detected_faces = []
        face_index = 0
        for x, y, w, h in faces:
            if w > 130:  # discard small detected faces

                face_detected = True
                if face_index == 0:
                    face_included_frames = (
                        face_included_frames + 1
                    )  # increase frame for a single face

                cv2.rectangle(
                    img, (x, y), (x + w, y + h), (67, 67, 67), 1
                )  # draw rectangle to main image

                cv2.putText(
                    img,
                    str(frame_threshold - face_included_frames),
                    (int(x + w / 4), int(y + h / 1.5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (255, 255, 255),
                    2,
                )

                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]  # crop detected face

                # -------------------------------------

                detected_faces.append((x, y, w, h))
                face_index = face_index + 1

                # -------------------------------------

        if face_detected == True and face_included_frames == frame_threshold and freeze == False:
            freeze = True
            # base_img = img.copy()
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()

        if freeze == True:

            toc = time.time()
            if (toc - tic) < time_threshold:

                if freezed_frame == 0:
                    freeze_img = base_img.copy()
                    # here, np.uint8 handles showing white area issue
                    # freeze_img = np.zeros(resolution, np.uint8)

                    for detected_face in detected_faces_final:
                        x = detected_face[0]
                        y = detected_face[1]
                        w = detected_face[2]
                        h = detected_face[3]

                        cv2.rectangle(
                            freeze_img, (x, y), (x + w, y + h), (67, 67, 67), 1
                        )  # draw rectangle to main image

                        # -------------------------------
                        # extract detected face
                        custom_face = base_img[y : y + h, x : x + w]
                        # -------------------------------
                        # facial attribute analysis

                        if enable_face_analysis == True:

                            demographies = DeepFace.analyze(
                                img_path=custom_face,
                                detector_backend=detector_backend,
                                enforce_detection=False,
                                silent=True,
                            )

                            if len(demographies) > 0:
                                # directly access 1st face cos img is extracted already
                                demography = demographies[0]

                                if enable_emotion:
                                    emotion = demography["emotion"]
                                    emotion_df = pd.DataFrame(
                                        emotion.items(), columns=["emotion", "score"]
                                    )
                                    emotion_df = emotion_df.sort_values(
                                        by=["score"], ascending=False
                                    ).reset_index(drop=True)

                                    # background of mood box

                                    # transparency
                                    overlay = freeze_img.copy()
                                    opacity = 0.4

                                    if x + w + pivot_img_size < resolution_x:
                                        # right
                                        cv2.rectangle(
                                            freeze_img
                                            # , (x+w,y+20)
                                            ,
                                            (x + w, y),
                                            (x + w + pivot_img_size, y + h),
                                            (64, 64, 64),
                                            cv2.FILLED,
                                        )

                                        cv2.addWeighted(
                                            overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img
                                        )

                                    elif x - pivot_img_size > 0:
                                        # left
                                        cv2.rectangle(
                                            freeze_img
                                            # , (x-pivot_img_size,y+20)
                                            ,
                                            (x - pivot_img_size, y),
                                            (x, y + h),
                                            (64, 64, 64),
                                            cv2.FILLED,
                                        )

                                        cv2.addWeighted(
                                            overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img
                                        )

                                    for index, instance in emotion_df.iterrows():
                                        current_emotion = instance["emotion"]
                                        emotion_label = f"{current_emotion} "
                                        emotion_score = instance["score"] / 100

                                        bar_x = 35  # this is the size if an emotion is 100%
                                        bar_x = int(bar_x * emotion_score)

                                        if x + w + pivot_img_size < resolution_x:

                                            text_location_y = y + 20 + (index + 1) * 20
                                            text_location_x = x + w

                                            if text_location_y < y + h:
                                                cv2.putText(
                                                    freeze_img,
                                                    emotion_label,
                                                    (text_location_x, text_location_y),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5,
                                                    (255, 255, 255),
                                                    1,
                                                )

                                                cv2.rectangle(
                                                    freeze_img,
                                                    (x + w + 70, y + 13 + (index + 1) * 20),
                                                    (
                                                        x + w + 70 + bar_x,
                                                        y + 13 + (index + 1) * 20 + 5,
                                                    ),
                                                    (255, 255, 255),
                                                    cv2.FILLED,
                                                )

                                        elif x - pivot_img_size > 0:

                                            text_location_y = y + 20 + (index + 1) * 20
                                            text_location_x = x - pivot_img_size

                                            if text_location_y <= y + h:
                                                cv2.putText(
                                                    freeze_img,
                                                    emotion_label,
                                                    (text_location_x, text_location_y),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5,
                                                    (255, 255, 255),
                                                    1,
                                                )

                                                cv2.rectangle(
                                                    freeze_img,
                                                    (
                                                        x - pivot_img_size + 70,
                                                        y + 13 + (index + 1) * 20,
                                                    ),
                                                    (
                                                        x - pivot_img_size + 70 + bar_x,
                                                        y + 13 + (index + 1) * 20 + 5,
                                                    ),
                                                    (255, 255, 255),
                                                    cv2.FILLED,
                                                )

                                if enable_age_gender:
                                    apparent_age = demography["age"]
                                    dominant_gender = demography["dominant_gender"]
                                    gender = "M" if dominant_gender == "Man" else "W"
                                    logger.debug(f"{apparent_age} years old {dominant_gender}")
                                    analysis_report = str(int(apparent_age)) + " " + gender

                                    # -------------------------------

                                    info_box_color = (46, 200, 255)

                                    # top
                                    if y - pivot_img_size + int(pivot_img_size / 5) > 0:

                                        triangle_coordinates = np.array(
                                            [
                                                (x + int(w / 2), y),
                                                (
                                                    x + int(w / 2) - int(w / 10),
                                                    y - int(pivot_img_size / 3),
                                                ),
                                                (
                                                    x + int(w / 2) + int(w / 10),
                                                    y - int(pivot_img_size / 3),
                                                ),
                                            ]
                                        )

                                        cv2.drawContours(
                                            freeze_img,
                                            [triangle_coordinates],
                                            0,
                                            info_box_color,
                                            -1,
                                        )

                                        cv2.rectangle(
                                            freeze_img,
                                            (
                                                x + int(w / 5),
                                                y - pivot_img_size + int(pivot_img_size / 5),
                                            ),
                                            (x + w - int(w / 5), y - int(pivot_img_size / 3)),
                                            info_box_color,
                                            cv2.FILLED,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            analysis_report,
                                            (x + int(w / 3.5), y - int(pivot_img_size / 2.1)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (0, 111, 255),
                                            2,
                                        )

                                    # bottom
                                    elif (
                                        y + h + pivot_img_size - int(pivot_img_size / 5)
                                        < resolution_y
                                    ):

                                        triangle_coordinates = np.array(
                                            [
                                                (x + int(w / 2), y + h),
                                                (
                                                    x + int(w / 2) - int(w / 10),
                                                    y + h + int(pivot_img_size / 3),
                                                ),
                                                (
                                                    x + int(w / 2) + int(w / 10),
                                                    y + h + int(pivot_img_size / 3),
                                                ),
                                            ]
                                        )

                                        cv2.drawContours(
                                            freeze_img,
                                            [triangle_coordinates],
                                            0,
                                            info_box_color,
                                            -1,
                                        )

                                        cv2.rectangle(
                                            freeze_img,
                                            (x + int(w / 5), y + h + int(pivot_img_size / 3)),
                                            (
                                                x + w - int(w / 5),
                                                y + h + pivot_img_size - int(pivot_img_size / 5),
                                            ),
                                            info_box_color,
                                            cv2.FILLED,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            analysis_report,
                                            (x + int(w / 3.5), y + h + int(pivot_img_size / 1.5)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (0, 111, 255),
                                            2,
                                        )

                        # --------------------------------
                        # face recognition
                        # call find function for custom_face

                        dfs = DeepFace.find(
                            img_path=custom_face,
                            db_path=db_path,
                            model_name=model_name,
                            detector_backend=detector_backend,
                            distance_metric=distance_metric,
                            enforce_detection=False,
                            silent=True,
                        )

                        if len(dfs) > 0:
                            # directly access 1st item because custom face is extracted already
                            df = dfs[0]

                            if df.shape[0] > 0:
                                candidate = df.iloc[0]
                                label = candidate["identity"]

                                # to use this source image as is
                                display_img = cv2.imread(label)
                                # to use extracted face
                                source_objs = DeepFace.extract_faces(
                                    img_path=label,
                                    target_size=(pivot_img_size, pivot_img_size),
                                    detector_backend=detector_backend,
                                    enforce_detection=False,
                                    align=False,
                                )

                                if len(source_objs) > 0:
                                    # extract 1st item directly
                                    source_obj = source_objs[0]
                                    display_img = source_obj["face"]
                                    display_img *= 255
                                    display_img = display_img[:, :, ::-1]
                                # --------------------
                                label = label.split("/")[-1]

                                try:
                                    if (
                                        y - pivot_img_size > 0
                                        and x + w + pivot_img_size < resolution_x
                                    ):
                                        # top right
                                        freeze_img[
                                            y - pivot_img_size : y,
                                            x + w : x + w + pivot_img_size,
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x + w, y),
                                            (x + w + pivot_img_size, y + 20),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x + w, y + 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y),
                                            (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                                            (x + w, y - int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )

                                    elif (
                                        y + h + pivot_img_size < resolution_y
                                        and x - pivot_img_size > 0
                                    ):
                                        # bottom left
                                        freeze_img[
                                            y + h : y + h + pivot_img_size,
                                            x - pivot_img_size : x,
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x - pivot_img_size, y + h - 20),
                                            (x, y + h),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x - pivot_img_size, y + h - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y + h),
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (x, y + h + int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )

                                    elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                                        # top left
                                        freeze_img[
                                            y - pivot_img_size : y, x - pivot_img_size : x
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x - pivot_img_size, y),
                                            (x, y + 20),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x - pivot_img_size, y + 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y),
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y - int(pivot_img_size / 2),
                                            ),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y - int(pivot_img_size / 2),
                                            ),
                                            (x, y - int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )

                                    elif (
                                        x + w + pivot_img_size < resolution_x
                                        and y + h + pivot_img_size < resolution_y
                                    ):
                                        # bottom righ
                                        freeze_img[
                                            y + h : y + h + pivot_img_size,
                                            x + w : x + w + pivot_img_size,
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x + w, y + h - 20),
                                            (x + w + pivot_img_size, y + h),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x + w, y + h - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y + h),
                                            (
                                                x + int(w / 2) + int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (
                                                x + int(w / 2) + int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (x + w, y + h + int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )
                                except Exception as err:  # pylint: disable=broad-except
                                    logger.error(str(err))

                        tic = time.time()  # in this way, freezed image can show 5 seconds

                        # -------------------------------

                time_left = int(time_threshold - (toc - tic) + 1)

                cv2.rectangle(freeze_img, (10, 10), (90, 50), (67, 67, 67), -10)
                cv2.putText(
                    freeze_img,
                    str(time_left),
                    (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow("img", freeze_img)

                freezed_frame = freezed_frame + 1
            else:
                face_detected = False
                face_included_frames = 0
                freeze = False
                freezed_frame = 0

        else:
            cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()

'''
dpg.create_context()

img = cv2.imread('./TestImages/TEST_PNG.png')
scaledimg = functions.scale_image(img, 1024, 500)
cv2.imshow('Output', scaledimg)
flatimg = functions.flatten_img(scaledimg)

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(scaledimg.shape[1], scaledimg.shape[0], flatimg, tag='texture_tag', format=dpg.mvFormat_Float_rgb)

with dpg.window(tag='expression_display', autosize=True):
    dpg.add_image('expression_image')

with dpg.window(tag='Control Window', no_close=True, autosize=True):
        dpg.add_text('Sensitivities')
        dpg.add_slider_float(label='Angry', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Disgust', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Fear', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Happy', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Sad', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Suprise', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Neutral', default_value=0.0, max_value=100.0, min_value=-100.0)

dpg.create_viewport(title='Window 1', width=1024, height=576)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window('image_tag', True)

with dpg.item_handler_registry(tag='#resize_handler'):
    dpg.add_item_resize_handler(callback=functions.window_resized)

dpg.start_dearpygui()

dpg.destroy_context()
'''

"""
cam = cv2.VideoCapture(0)

_, img = cam.read()



while dpg.is_dearpygui_running():
    _, img = cam.read()

    data = numpy.flip(img, 2)
    data = data.ravel()
    data = numpy.asfarray(data, dtype='f')
    texture_data = numpy.true_divide(data, 255.0)
    dpg.set_value("texture_tag", texture_data)

    dpg.render_dearpygui_frame()

dpg.destroy_context()
"""

""" #
sensitivities = {
    "angry": 0.5,
    "disgust": 1.0,
    "fear": 1.0,
    "happy": 5.0,
    "sad": 0.5,
    "surprise": 1.0,
    "neutral": 1.5,
}


class device:
    def __init__(self, index, name):
        self.index = index
        self.name = name

def list_cams():
    devices = FilterGraph().get_input_devices()
    webcams = []
    for device_index, device_name in enumerate(devices):
        temp_device = device(device_index, device_name)
        webcams.append(temp_device)
    return webcams

list_cams = list_cams()
for d in list_cams:
    print("Index: %i Name: %s" % (d.index, d.name))
"""

""" #
expression_tracking.scan_for_cameras()

expression_tracking.begin_analysis(source = 0)
"""

"""
#
expression_tracking.analysis(db_path = "TestImages",
    model_name = "VGG-Face",
    detector_backend = "opencv",
    distance_metric = "cosine",
    enable_face_analysis  = True,
    source = 0,
    time_threshold = 0.5,
    frame_threshold = 1)

"""

'''for key in list(multipliers.keys()):
        multipliers[key] = dpg.get_value(key)
        if (multipliers[key] > 0): 
            demographies[key] = round(demographies[key] * multipliers[key], 2)
        elif (multipliers[key] < 0):
            demographies[key] = round(demographies[key] / multipliers[key], 2)
        else:
            demographies[key] = round(demographies[key], 2)''' 